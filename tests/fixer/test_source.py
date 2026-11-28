"""Unit tests for fixer source token slicing, comments/pragmas, and import manipulation."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from pydoppelgangerhunt import (
    analyze_unit_variable_scope,
    extract_unit_source_code,
    replace_unit_in_source,
    synthesize_shared_helper_code,
)
from pydoppelgangerhunt.fixer.source import (  # pylint: disable=protected-access
    _detect_indent_step,
    _extract_unit_body_lines,
    _find_module_helper_insertion_index,
    _get_module_imported_names,
    _insert_imports_into_module,
    find_enclosing_function_is_async,
)
from pydoppelgangerhunt.fixer._synthesis_types import (  # pylint: disable=protected-access
    _extract_required_typing_imports,
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
    scope_inner = analyze_unit_variable_scope(u_inner, repo_root=str(tmp_path))
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
    scope_ro = analyze_unit_variable_scope(u_ro, repo_root=str(tmp_path))
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
    helper_gen_from = synthesize_shared_helper_code(
        u_gen_from, u_gen_from, include_imports=True, repo_root=str(tmp_path)
    )
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
    helper_gen_val = synthesize_shared_helper_code(u_gen_val, u_gen_val, repo_root=str(tmp_path))
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


def test_resolve_safe_unit_file_path_symlinked_root_prefix(tmp_path: Path) -> None:
    """Verifies that _resolve_safe_unit_file_path handles absolute paths with symlink roots."""
    from pydoppelgangerhunt.fixer._unit_schema import _resolve_safe_unit_file_path

    real_repo = tmp_path / "real_repo"
    real_repo.mkdir()
    target_py = real_repo / "main.py"
    target_py.write_text("print('hello')\n", encoding="utf-8")

    # Direct resolution under real root
    resolved = _resolve_safe_unit_file_path({"file": str(target_py)}, repo_root=str(real_repo))
    assert resolved == target_py.resolve()

    # Outside root should return None
    outside_py = tmp_path / "outside.py"
    outside_py.write_text("print('outside')\n", encoding="utf-8")
    assert _resolve_safe_unit_file_path({"file": str(outside_py)}, repo_root=str(real_repo)) is None

    # Platform symlinked root alias test (if symlinks supported on OS/privilege)
    sym_repo = tmp_path / "sym_repo"
    try:
        sym_repo.symlink_to(real_repo, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks not supported on this platform/privilege")

    # When file path uses sym_repo prefix and repo_root uses real_repo (or vice versa)
    res1 = _resolve_safe_unit_file_path(
        {"file": str(sym_repo / "main.py")}, repo_root=str(real_repo)
    )
    assert res1 == target_py.resolve()

    res2 = _resolve_safe_unit_file_path(
        {"file": str(target_py)}, repo_root=str(sym_repo)
    )
    assert res2 == target_py.resolve()

    # Symlinked file inside the repository must be rejected
    sym_file = real_repo / "sym_link.py"
    try:
        sym_file.symlink_to(target_py)
    except (OSError, NotImplementedError):
        pass
    else:
        assert _resolve_safe_unit_file_path(
            {"file": str(sym_file)}, repo_root=str(real_repo)
        ) is None

    # Symlinked directory inside the repository must also be rejected
    sym_dir = real_repo / "sym_subdir"
    try:
        sym_dir.symlink_to(real_repo, target_is_directory=True)
    except (OSError, NotImplementedError):
        pass
    else:
        assert _resolve_safe_unit_file_path(
            {"file": str(sym_dir / "main.py")}, repo_root=str(real_repo)
        ) is None

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
    from pydoppelgangerhunt.fixer.source import _insert_imports_into_module  # pylint: disable=import-outside-toplevel

    orig = ["def foo(): pass\n"]
    imports = ["from typing import Any", "from typing import Any", "from typing import Tuple"]
    res = _insert_imports_into_module(orig, imports)
    assert res.count("from typing import Any\n") == 1
    assert res.count("from typing import Tuple\n") == 1

def test_extract_unit_body_lines_and_scope_method_kind(tmp_path: Path) -> None:
    """Verifies that units with kind='method' strip headers/docstrings and preserve method parameters."""
    from pydoppelgangerhunt.fixer.source import _extract_unit_body_lines  # pylint: disable=import-outside-toplevel

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
    from pydoppelgangerhunt.fixer.source import _find_module_helper_insertion_index  # pylint: disable=import-outside-toplevel

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
    from pydoppelgangerhunt.fixer.source import (  # pylint: disable=import-outside-toplevel
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

def test_get_module_imported_names_guarded_imports() -> None:
    """Verifies that _get_module_imported_names inspects top-level If and Try import blocks."""
    from pydoppelgangerhunt.fixer.source import _get_module_imported_names  # pylint: disable=import-outside-toplevel

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

def test_slice_unit_token_lines_single_line_bounds() -> None:
    """Verifies defensive single-line column bounds check in _slice_unit_token_lines."""
    from pydoppelgangerhunt.fixer.source import _slice_unit_token_lines  # pylint: disable=import-outside-toplevel

    line = ["    total = sum(x for x in data)"]
    # Normal column slice
    res_normal = _slice_unit_token_lines(
        {"kind": "comprehension", "start_col": 12, "end_col": 32},
        list(line),
    )
    assert res_normal == ["sum(x for x in data)"]

    # Inverted column bounds (e_col < s_col) must fail closed by returning None rather than slicing to EOL
    res_inverted = _slice_unit_token_lines(
        {"kind": "comprehension", "start_col": 12, "end_col": 5},
        list(line),
    )
    assert res_inverted is None


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


def test_extract_required_typing_imports_corrupt_source_fallback() -> None:
    """Verifies that _extract_required_typing_imports handles ValueError from null bytes gracefully."""
    corrupt_sig = "x: List[str]\x00 -> None"
    # Should fall back to regex token matching and find List
    res = _extract_required_typing_imports(corrupt_sig)
    assert "List" in res


def test_find_enclosing_ast_node_corrupt_source_fallback() -> None:
    """Verifies that _find_innermost_enclosing_node handles ValueError from null bytes gracefully."""
    from pydoppelgangerhunt.fixer.binding import (  # pylint: disable=import-outside-toplevel
        _find_innermost_enclosing_node,
    )

    u = {"file": "test.py", "start": 1, "end": 2, "name": "foo", "kind": "function"}
    res = _find_innermost_enclosing_node("def foo():\n    pass\x00", u, (ast.FunctionDef,))
    assert res is None


def test_find_module_helper_insertion_index_with_suppress() -> None:
    """Verifies that helpers are placed after 'with suppress(ImportError): import ...' blocks."""
    lines = [
        "from contextlib import suppress\n",
        "with suppress(ImportError):\n",
        "    import optional_dep\n",
        "\n",
        "def first_func():\n",
        "    pass\n",
    ]
    idx = _find_module_helper_insertion_index(lines)
    assert idx >= 3

    src = "".join(lines)
    names = _get_module_imported_names(src, include_conditional=True)
    assert "optional_dep" in names


def test_find_enclosing_function_is_async_decorated() -> None:
    """Verifies that find_enclosing_function_is_async recognizes decorated async functions
    when the unit start line begins on or after the decorator."""
    src = (
        "@deco1\n"
        "@deco2(arg=True)\n"
        "async def async_worker(x):\n"
        "    res = x * 2\n"
        "    return res\n"
    )
    # Unit starting on the decorator itself
    assert find_enclosing_function_is_async(src, start_line=1, end_line=5) is True
    # Unit starting on second decorator
    assert find_enclosing_function_is_async(src, start_line=2, end_line=4) is True
    # Unit inside function body
    assert find_enclosing_function_is_async(src, start_line=4, end_line=4) is True
    # Out of range
    assert find_enclosing_function_is_async(src, start_line=6, end_line=7) is False

    # Sync decorated function should return False
    sync_src = (
        "@deco\n"
        "def sync_worker(x):\n"
        "    return x * 2\n"
    )
    assert find_enclosing_function_is_async(sync_src, start_line=1, end_line=3) is False


def test_find_enclosing_function_is_async_nested_tiebreaker() -> None:
    """Verifies that when multiple candidate functions enclose a unit, candidates are
    sorted primarily by span and tie-broken by -earliest_start to select the innermost function."""
    # Outer sync function enclosing inner async function
    src_sync_outer = (
        "def outer():\n"
        "    async def inner():\n"
        "        yield 1\n"
        "    return inner\n"
    )
    # Unit inside inner (line 3) -> should resolve to inner's async status (True)
    assert find_enclosing_function_is_async(src_sync_outer, start_line=3, end_line=3) is True

    # Outer async function enclosing inner sync function
    src_async_outer = (
        "async def outer():\n"
        "    def inner():\n"
        "        return 1\n"
        "    return inner()\n"
    )
    # Unit inside inner (line 3) -> should resolve to inner's sync status (False)
    assert find_enclosing_function_is_async(src_async_outer, start_line=3, end_line=3) is False


def test_is_subroutine_unit_comprehension_with_source() -> None:
    """Verifies that comprehension, closure, and complex_expr units return False
    from is_subroutine_unit even when full source text is provided."""
    from pydoppelgangerhunt.fixer.source import is_subroutine_unit

    src = (
        "def compute(data):\n"
        "    res = [x * 2 for x in data]\n"
        "    return res\n"
    )
    tree = ast.parse(src)
    comp_unit = {
        "file": "test.py",
        "kind": "comprehension",
        "start": 2,
        "end": 2,
        "start_col": 10,
        "end_col": 31,
    }
    # With source and tree provided, comprehension must NOT fall through to structural check
    assert is_subroutine_unit(comp_unit, source_text=src, tree=tree) is False

    closure_unit = {
        "file": "test.py",
        "kind": "closure",
        "start": 2,
        "end": 2,
    }
    assert is_subroutine_unit(closure_unit, source_text=src, tree=tree) is False

    expr_unit = {
        "file": "test.py",
        "kind": "complex_expr",
        "start": 2,
        "end": 2,
    }
    assert is_subroutine_unit(expr_unit, source_text=src, tree=tree) is False

    class_unit = {
        "file": "test.py",
        "kind": "class",
        "start": 2,
        "end": 3,
    }
    assert is_subroutine_unit(class_unit, source_text=src, tree=tree) is False

    data_table_unit = {
        "file": "test.py",
        "kind": "data_table",
        "start": 2,
        "end": 2,
    }
    assert is_subroutine_unit(data_table_unit, source_text=src, tree=tree) is False


def test_slice_unit_token_lines_invalid_or_negative_start_col() -> None:
    """Verifies that _slice_unit_token_lines fails closed (returns None) on negative or non-numeric column offsets."""
    from pydoppelgangerhunt.fixer.source import _slice_unit_token_lines

    lines = ["    res = [x for x in data]\n"]
    # Negative start_col and negative end_col both fail closed, even on col-sliced buffers.
    for bad_unit in (
        {"kind": "comprehension", "start_col": -5, "end_col": 10},
        {"kind": "comprehension", "start_col": 4, "end_col": -2},
        {
            "kind": "comprehension",
            "start_col": -5,
            "end_col": 10,
            "source_lines_is_col_sliced": True,
        },
        {
            "kind": "comprehension",
            "start_col": 4,
            "end_col": -2,
            "source_lines_is_col_sliced": True,
        },
        {"kind": "comprehension", "start_col": "invalid", "end_col": 10},
    ):
        assert _slice_unit_token_lines(bad_unit, lines) is None


def test_is_sliced_excerpt_for_file_starts_at_line_one() -> None:
    """Verifies that _is_sliced_excerpt_for_file identifies an excerpt starting at line 1."""
    from pydoppelgangerhunt.fixer.source import _is_sliced_excerpt_for_file

    u = {"start": 1, "end": 3}
    lines = ["a = 1\n", "b = 2\n", "c = 3\n"]
    assert _is_sliced_excerpt_for_file(u, lines) is True


def test_load_unit_file_text_prioritizes_disk_over_source_lines(tmp_path: Path) -> None:
    """Verifies that _load_unit_file_text prioritizes disk file over in-memory source_lines."""
    from pydoppelgangerhunt.fixer.source import _load_unit_file_text

    full_text = "x = 1\ny = 2\nz = 3\nw = 4\nv = 5\n"
    target_file = tmp_path / "module.py"
    target_file.write_text(full_text, encoding="utf-8")

    unit = {
        "file": str(target_file),
        "start": 1,
        "end": 2,
        "source_lines": ["x = 1\n", "y = 2\n"],
    }
    loaded = _load_unit_file_text(unit, repo_root=str(tmp_path))
    assert loaded == full_text


def test_dedent_source_text_multiple_form_feeds() -> None:
    """Verifies that dedent_source_text correctly preserves multiple form feed characters."""
    from pydoppelgangerhunt.fixer.source import dedent_source_text

    raw = " a\n\x0c b\n\x0c c\n"
    dedented = dedent_source_text(raw)
    assert dedented == "a\n\x0cb\n\x0cc\n"


def test_load_unit_file_text_prioritizes_unsliced_source_lines_over_disk(tmp_path: Path) -> None:
    """Verifies that unsliced in-memory source_lines take precedence over disk (IDE buffer)."""
    from pydoppelgangerhunt.fixer.source import _load_unit_file_text

    disk_text = "x = 1\ny = 2\n"
    target_file = tmp_path / "module.py"
    target_file.write_text(disk_text, encoding="utf-8")

    unsaved_lines = ["x = 100\n", "y = 200\n", "z = 300\n"]
    unit = {
        "file": str(target_file),
        "start": 1,
        "end": 2,
        "source_lines": unsaved_lines,
    }
    loaded = _load_unit_file_text(unit, repo_root=str(tmp_path))
    assert loaded == "x = 100\ny = 200\nz = 300\n"

    empty_unsaved = {
        "file": str(target_file),
        "start": 1,
        "end": 2,
        "source_lines": [],
        "source_lines_is_sliced": False,
    }
    assert _load_unit_file_text(empty_unsaved, repo_root=str(tmp_path)) == ""


def test_slice_unit_token_lines_offset_aliases() -> None:
    """Verifies that _slice_unit_token_lines respects start_col_offset, col_offset, and end_col_offset."""
    from pydoppelgangerhunt.fixer.source import _slice_unit_token_lines

    lines = ["    total = [x * 2 for x in items]\n"]
    u_aliases = {
        "kind": "comprehension",
        "start_col_offset": 12,
        "end_col_offset": 34,
    }
    sliced = _slice_unit_token_lines(u_aliases, list(lines))
    assert sliced == ["[x * 2 for x in items]"]

    u_col_offset = {
        "kind": "comprehension",
        "col_offset": 12,
        "end_col": 34,
    }
    assert _slice_unit_token_lines(u_col_offset, list(lines)) == [
        "[x * 2 for x in items]"
    ]

    # Multi-line unit with offset aliases
    multi_lines = [
        "    result = (\n",
        "        [x * 2 for x in items]\n",
        "    )\n",
    ]
    u_multi = {
        "kind": "complex_expr",
        "start_col_offset": 13,
        "end_col_offset": 5,
    }
    sliced_multi = _slice_unit_token_lines(u_multi, list(multi_lines))
    assert sliced_multi is not None
    assert sliced_multi[0] == "(\n"
    assert sliced_multi[1] == "        [x * 2 for x in items]\n"
    assert sliced_multi[2] == "    )"


def test_is_sliced_excerpt_for_file_explicit_flag_and_ambiguous_span() -> None:
    """Verifies that _is_sliced_excerpt_for_file honors source_lines_is_sliced: False
    for full-file buffers whose line count matches the unit span."""
    from pydoppelgangerhunt.fixer.source import _is_sliced_excerpt_for_file

    lines = ["x = 1\n", "y = 2\n", "z = 3\n"]
    unit_base = {"file": "m.py", "start": 1, "end": 3}

    # 1. When source_lines_is_sliced is False, it is recognized as full file text
    unit_full = dict(unit_base, source_lines_is_sliced=False)
    assert not _is_sliced_excerpt_for_file(unit_full, lines)

    # 2. When source_lines_is_sliced is omitted, length heuristic fails closed (ambiguous)
    unit_unspecified = dict(unit_base)
    assert _is_sliced_excerpt_for_file(unit_unspecified, lines)

    # 3. When source_lines_is_sliced is True, it is recognized as sliced
    unit_sliced = dict(unit_base, source_lines_is_sliced=True)
    assert _is_sliced_excerpt_for_file(unit_sliced, lines)

    # 4. Truncated buffer (len(lines) < u_end) is always treated as sliced
    assert _is_sliced_excerpt_for_file(unit_full, lines[:2])


def test_extract_required_typing_imports_generator_types() -> None:
    """Verifies that Generator and AsyncGenerator are extracted from type annotations."""
    from pydoppelgangerhunt.fixer._synthesis_types import (  # pylint: disable=import-outside-toplevel
        _extract_required_typing_imports,
    )

    sync_sig = "def gen() -> Generator[int, Any, None]: pass"
    extracted_sync = _extract_required_typing_imports(sync_sig)
    assert "Generator" in extracted_sync
    assert "Any" in extracted_sync

    async_sig = "def agen() -> AsyncGenerator[str, Any]: pass"
    extracted_async = _extract_required_typing_imports(async_sig)
    assert "AsyncGenerator" in extracted_async
    assert "Any" in extracted_async


def test_extract_unit_body_lines_indented_method_col0_multiline_string() -> None:
    """Verifies that _extract_unit_body_lines strips the method signature and docstring
    and dedents body statements to column 0 when an indented method contains a multiline
    string or docstring with continuation lines at column 0."""
    raw_lines = [
        "    def format_banner(self, title: str) -> str:",
        '        """Docstring header',
        "col_0_docstring_line",
        '        """',
        '        banner = f"""header {',
        "title.strip()",
        '}"""',
        "        clean = title.strip()",
        "        return banner + clean",
    ]
    body_lines = _extract_unit_body_lines({"kind": "method"}, raw_lines)
    assert body_lines == [
        'banner = f"""header {',
        "title.strip()",
        '}"""',
        "clean = title.strip()",
        "return banner + clean",
    ]

    from pydoppelgangerhunt.fixer.source import (  # pylint: disable=import-outside-toplevel
        _find_multiline_string_continuation_lines,
        dedent_source_text,
    )

    cr_text = "    x = '''header\rcol0_line'''\r    y = 1\r"
    assert _find_multiline_string_continuation_lines(cr_text) == {2}
    assert dedent_source_text(cr_text) == "x = '''header\rcol0_line'''\ry = 1\r"


@pytest.mark.parametrize(
    ("case_id", "src", "unit_template"),
    [
        pytest.param(
            "utf8_multibyte_same_line_bytes_vs_chars",
            "# header café\nval = 'αβ🚀' + target_expr\n",
            {"start": 2, "end": 2, "kind": "complex_expr"},
            id="utf8_multibyte_same_line",
        ),
        pytest.param(
            "crlf_line_endings_preserved_across_lines",
            "x = 1\r\nres = 'π' + target_expr\r\nz = 3\r\n",
            {"start": 2, "end": 2, "kind": "complex_expr"},
            id="crlf_line_endings",
        ),
        pytest.param(
            "lone_cr_line_endings_preserved_across_lines",
            "x = 1\rres = 'é' + target_expr\rz = 3\r",
            {"start": 2, "end": 2, "kind": "complex_expr"},
            id="lone_cr_line_endings",
        ),
    ],
)
def test_utf8_byte_and_char_offsets_and_line_ending_preservation(
    case_id: str,
    src: str,
    unit_template: dict[str, object],
) -> None:
    """Verifies UTF-8 byte vs character offsets and CRLF/lone-CR replacement span integrity."""
    from pydoppelgangerhunt import compute_unit_byte_offsets, compute_unit_char_offsets

    expr = "target_expr"
    start_val = unit_template["start"]
    assert isinstance(start_val, int)
    target_line_idx = start_val - 1
    from pydoppelgangerhunt.source_lines import split_source_lines

    line_str = split_source_lines(src)[target_line_idx]
    char_start = line_str.index(expr)
    byte_start = len(line_str[:char_start].encode("utf-8"))
    byte_end = byte_start + len(expr.encode("utf-8"))
    unit = {
        **unit_template,
        "start_col": byte_start,
        "end_col": byte_end,
        "source_text": src,
    }
    sb, eb = compute_unit_byte_offsets(src, unit)
    sc, ec = compute_unit_char_offsets(src, unit)
    assert src.encode("utf-8")[sb:eb].decode("utf-8") == expr, case_id
    assert src[sc:ec] == expr, case_id
    replaced = replace_unit_in_source(src, unit, "replacement_call()")
    assert replaced.replace("replacement_call()", expr) == src, case_id


def test_column_slicing_exact_eof_col_oob_col_and_no_double_slicing() -> None:
    """Verifies exact end_col==len, end_col==len+1 rejection, and source_lines_is_col_sliced non-double-slicing."""
    from pydoppelgangerhunt.fixer.scope import _has_out_of_bounds_unit_cols
    from pydoppelgangerhunt.fixer.source import _slice_unit_token_lines

    full_line = "val = x + y\n"
    code_bytes = len(full_line.rstrip("\n").encode("utf-8"))  # 11

    # Exact end_col == code_bytes succeeds cleanly
    sliced_exact = _slice_unit_token_lines(
        {"kind": "complex_expr", "start": 1, "end": 1, "start_col": 6, "end_col": code_bytes},
        [full_line],
    )
    assert sliced_exact == ["x + y"]

    # end_col == code_bytes + 1 fails closed in both _slice_unit_token_lines and _has_out_of_bounds_unit_cols
    assert (
        _slice_unit_token_lines(
            {"kind": "complex_expr", "start": 1, "end": 1, "start_col": 6, "end_col": code_bytes + 1},
            [full_line],
        )
        is None
    )
    assert (
        _has_out_of_bounds_unit_cols(
            {"kind": "complex_expr", "start": 1, "end": 1, "start_col": 6, "end_col": code_bytes + 1},
            [full_line],
            is_placeholder_origin=False,
            explicit_col_sliced=False,
        )
        is True
    )

    # source_lines_is_col_sliced=True prevents double slicing already trimmed excerpts
    already_trimmed = ["x + y"]
    no_reslice = _slice_unit_token_lines(
        {
            "kind": "complex_expr",
            "start": 1,
            "end": 1,
            "start_col": 6,
            "end_col": code_bytes,
            "source_lines_is_col_sliced": True,
        },
        already_trimmed,
    )
    assert no_reslice == already_trimmed


def test_col_offset_alias_across_slicing_normalization_synthesis_and_patch(
    tmp_path: Path,
) -> None:
    """Verifies 'col_offset' acts consistently as 'start_col' across slicing, validation, synthesis, and patching."""
    from pydoppelgangerhunt.fixer.patch import generate_refactoring_patch
    from pydoppelgangerhunt.fixer.source import (
        _slice_unit_token_lines,
        validate_unit_schema,
    )
    from pydoppelgangerhunt.fixer.synthesis import synthesize_shared_helper_code
    from pydoppelgangerhunt.source_lines import (
        normalize_unit_coordinates,
        parse_unit_coord,
    )

    line_text = "result = (alpha + beta) * 2\n"
    expr = "(alpha + beta) * 2"
    col_start = line_text.index(expr)
    col_end = col_start + len(expr)

    # Direct token slicing respects col_offset and strips leading assignment prefix
    sliced = _slice_unit_token_lines(
        {
            "kind": "complex_expr",
            "start": 1,
            "end": 1,
            "col_offset": col_start,
            "end_col": col_end,
        },
        [line_text],
    )
    assert sliced == [expr]

    norm_u = normalize_unit_coordinates(
        {"start": 1, "end": 1, "col_offset": col_start, "end_col_offset": col_end}
    )
    assert norm_u["start_col"] == col_start
    assert norm_u["end_col"] == col_end
    assert parse_unit_coord({"col_offset": col_start}, "start_col", default=None) == col_start

    # Conflicting start_col vs col_offset fails validation cleanly
    with pytest.raises(ValueError, match="Conflicting coordinate aliases"):
        validate_unit_schema(
            {"start": 1, "end": 1, "start_col": col_start + 1, "col_offset": col_start}
        )

    # End-to-end synthesis & patch generation with col_offset-only expression units
    src = (
        "def fn_one(a, b):\n"
        "    out = (a + b) * 2 + 1\n"
        "    return out\n"
        "\n"
        "def fn_two(c, d):\n"
        "    val = (c + d) * 2 + 1\n"
        "    return val\n"
    )
    mod_file = tmp_path / "expr_mod.py"
    mod_file.write_text(src, encoding="utf-8")
    lines = src.splitlines(keepends=True)
    e1 = "(a + b) * 2 + 1"
    e2 = "(c + d) * 2 + 1"
    u1: dict[str, Any] = {
        "file": str(mod_file),
        "name": "fn_one",
        "kind": "complex_expr",
        "start": 2,
        "end": 2,
        "col_offset": lines[1].index(e1),
        "end_col": lines[1].index(e1) + len(e1),
    }
    u2: dict[str, Any] = {
        "file": str(mod_file),
        "name": "fn_two",
        "kind": "complex_expr",
        "start": 6,
        "end": 6,
        "col_offset": lines[5].index(e2),
        "end_col": lines[5].index(e2) + len(e2),
    }

    helper = synthesize_shared_helper_code(u1, u2, repo_root=str(tmp_path))
    assert "def _shared_" in helper
    assert "return " in helper
    ast.parse(helper)

    patch_text = generate_refactoring_patch(
        [(1.0, u1, u2)], repo_root=str(tmp_path), replace_clones=True
    )
    assert "def _shared_" in patch_text
    assert "_shared_" in patch_text
    assert "out = _shared_" in patch_text
    assert "val = _shared_" in patch_text


@pytest.mark.parametrize(
    ("line", "start_col", "end_col", "expected"),
    [
        pytest.param("x = [i for i in items]", 4, 4, None, id="zero_width_interior_rejected"),
        pytest.param("x = [i for i in items]", 0, 0, None, id="zero_width_origin_rejected"),
        pytest.param("x = [i for i in items]", 22, 22, None, id="zero_width_eol_rejected"),
        pytest.param("x = [i for i in items]", None, 0, None, id="implicit_zero_to_zero_rejected"),
        pytest.param("", 0, 0, None, id="empty_line_explicit_zero_width_rejected"),
        pytest.param("", None, None, [""], id="control_empty_line_without_cols_allowed"),
        pytest.param("x = [i for i in items]", 4, 22, ["[i for i in items]"], id="control_valid_span_sliced"),
    ],
)
def test_slice_unit_token_lines_rejects_zero_width_single_line_spans(
    line: str,
    start_col: int | None,
    end_col: int | None,
    expected: list[str] | None,
) -> None:
    """Verifies _slice_unit_token_lines rejects zero-width single-line spans consistently with schema validation."""
    from pydoppelgangerhunt.fixer.source import (  # pylint: disable=import-outside-toplevel
        _slice_unit_token_lines,
    )

    unit: dict[str, Any] = {"kind": "comprehension", "start": 1, "end": 1}
    if start_col is not None:
        unit["start_col"] = start_col
    if end_col is not None:
        unit["end_col"] = end_col
    assert _slice_unit_token_lines(unit, [line]) == expected


@pytest.mark.parametrize(
    ("source_lines", "is_sliced_flag", "expect_from_lines"),
    [
        pytest.param(
            ["def from_lines():\n", "    return 1\n", "def extra():\n", "    return 2\n"],
            None,
            True,
            id="non_sliced_source_lines_beats_conflicting_source_text",
        ),
        pytest.param(
            ["def from_lines():\n", "    return 1\n"],
            False,
            True,
            id="explicit_unsliced_source_lines_beats_conflicting_source_text",
        ),
        pytest.param(
            ["def from_lines():\n", "    return 1\n"],
            True,
            False,
            id="control_pre_sliced_source_lines_falls_through_to_full_source_text",
        ),
        pytest.param(
            [],
            None,
            False,
            id="control_empty_source_lines_falls_through_to_source_text",
        ),
    ],
)
def test_load_unit_file_text_matches_extract_source_lines_precedence_on_py_units(
    source_lines: list[str],
    is_sliced_flag: bool | None,
    expect_from_lines: bool,
) -> None:
    """Verifies _load_unit_file_text prioritizes non-sliced source_lines ahead of source_text on .py units."""
    from pydoppelgangerhunt.fixer.source import (  # pylint: disable=import-outside-toplevel
        _load_unit_file_text,
    )
    from pydoppelgangerhunt.source_lines import (  # pylint: disable=import-outside-toplevel
        extract_unit_source_lines_with_state,
    )

    source_text = "def from_text():\n    return 99\n"
    unit: dict[str, Any] = {
        "file": "mod.py",
        "name": "target",
        "start": 1,
        "end": 2,
        "source_lines": source_lines,
        "source_text": source_text,
    }
    if is_sliced_flag is not None:
        unit["source_lines_is_sliced"] = is_sliced_flag

    loaded = _load_unit_file_text(unit)
    extracted_lines, _ = extract_unit_source_lines_with_state(unit)
    if expect_from_lines:
        assert loaded == "".join(source_lines)
        assert "".join(extracted_lines) == "".join(source_lines[:2])
    else:
        assert loaded == source_text


@pytest.mark.parametrize(
    ("unit_overrides", "expect_safe"),
    [
        pytest.param(
            {"file": "nb.ipynb#bad", "kind": "function"},
            False,
            id="malformed_ipynb_fragment_with_source_code",
        ),
        pytest.param(
            {"file": "nb.ipynb#cell_0", "kind": "function"},
            False,
            id="zero_indexed_ipynb_cell_anchor_with_source_code",
        ),
        pytest.param(
            {
                "file": "nb.ipynb#cell_99",
                "kind": "function",
                "file_source": (
                    '{"cells": [{"cell_type": "code", "source": ["x = 1\\n"]}], '
                    '"metadata": {}, "nbformat": 4, "nbformat_minor": 5}'
                ),
            },
            False,
            id="out_of_range_nbformat_json_cell_with_source_code",
        ),
        pytest.param(
            {
                "file": "nb.ipynb#cell_2",
                "kind": "function",
                "source_text": (
                    '{"cells": ['
                    '{"cell_type": "code", "source": ["x = 1\\n"]}, '
                    '{"cell_type": "markdown", "source": ["# Heading\\n"]}'
                    '], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}'
                ),
            },
            False,
            id="markdown_nbformat_json_cell_with_source_code",
        ),
        pytest.param(
            {
                "file": "nb.ipynb#cell_1",
                "kind": "basic_block",
                "is_subroutine": True,
                "allow_snippet_only": True,
                "file_source": "{corrupt json",
            },
            False,
            id="corrupt_nbformat_json_subroutine_even_with_allow_snippet_only",
        ),
        pytest.param(
            {"file": "virtual/mod.py", "kind": "function"},
            True,
            id="control_virtual_py_snippet_fallback_stays_allowed",
        ),
        pytest.param(
            {
                "file": "nb.ipynb#cell_1",
                "kind": "function",
                "source_text": "def f(x):\n    return x + 1\n",
            },
            True,
            id="control_valid_plain_python_notebook_cell_buffer",
        ),
    ],
)
def test_inspect_single_unit_scope_fails_closed_on_invalid_notebook_anchors_with_source_code(
    tmp_path: Path,
    unit_overrides: dict[str, Any],
    expect_safe: bool,
) -> None:
    """Verifies invalid .ipynb anchors/buffers fail closed in inspect_single_unit_scope even with source_code."""
    from pydoppelgangerhunt.fixer.scope import (  # pylint: disable=import-outside-toplevel
        inspect_single_unit_scope,
    )

    unit: dict[str, Any] = {
        "name": "f",
        "start": 1,
        "end": 2,
        "source_code": "def f(x):\n    return x + 1\n",
        **unit_overrides,
    }
    scope = inspect_single_unit_scope(unit, repo_root=str(tmp_path))
    assert scope["is_control_flow_safe"] is expect_safe, scope
    if not expect_safe:
        assert "unavailable_source_placeholder" in scope["control_flow_hazards"]
    else:
        assert "unavailable_source_placeholder" not in scope["control_flow_hazards"]

    # Unanchored .ipynb file on disk must also fail closed even when source_code is populated
    raw_nb = tmp_path / "unanchored.ipynb"
    raw_nb.write_text(
        '{"cells": [{"cell_type": "code", "source": ["def f(x):\\n", "    return x + 1\\n"]}], '
        '"metadata": {}, "nbformat": 4, "nbformat_minor": 5}',
        encoding="utf-8",
    )
    unanchored_scope = inspect_single_unit_scope(
        {
            "file": str(raw_nb),
            "name": "f",
            "start": 1,
            "end": 2,
            "kind": "function",
            "source_code": "def f(x):\n    return x + 1\n",
        },
        repo_root=str(tmp_path),
    )
    assert unanchored_scope["is_control_flow_safe"] is False
    assert "unavailable_source_placeholder" in unanchored_scope["control_flow_hazards"]


@pytest.mark.parametrize(
    (
        "kind",
        "source_text",
        "start_line",
        "end_line",
        "full_lines",
        "sliced_lines",
        "start_col",
        "end_col",
        "expected_inputs",
        "expected_stores",
        "expect_safe",
    ),
    [
        pytest.param(
            "basic_block",
            (
                "def work(c, e, g):\n"
                "    a = 1; b = c + 1; d = e + 2\n"
                "    f = g + 3; h = 4\n"
                "    return b, d, f\n"
            ),
            2,
            3,
            ["    a = 1; b = c + 1; d = e + 2\n", "    f = g + 3; h = 4\n"],
            ["    b = c + 1; d = e + 2\n", "    f = g + 3\n"],
            11,
            13,
            ["c", "e", "g"],
            ["b", "d", "f"],
            True,
            id="multiline_semicolon_basic_block_pre_col_sliced_not_double_sliced",
        ),
        pytest.param(
            "basic_block",
            None,
            2,
            2,
            ["    b = c + 1; d = e + 2  # ok\n"],
            ["b = c + 1; d = e + 2  # ok\n"],
            4,
            20,
            ["c", "e"],
            ["b", "d"],
            True,
            id="single_line_semicolon_basic_block_pre_col_sliced_not_double_sliced",
        ),
        pytest.param(
            "closure",
            None,
            2,
            2,
            ["    fn =  lambda x: lambda y: x + y + offset\n"],
            ["lambda x: lambda y: x + y + offset\n"],
            10,
            34,
            ["offset"],
            [],
            False,
            id="single_line_nested_lambda_closure_pre_col_sliced_not_double_sliced",
        ),
    ],
)
def test_inspect_single_unit_scope_does_not_double_slice_when_col_sliced_flag_true(
    kind: str,
    source_text: str | None,
    start_line: int,
    end_line: int,
    full_lines: list[str],
    sliced_lines: list[str],
    start_col: int,
    end_col: int,
    expected_inputs: list[str],
    expected_stores: list[str],
    expect_safe: bool,
) -> None:
    """Verifies non-expression units with source_lines_is_col_sliced=True are not sliced a second time."""
    from pydoppelgangerhunt.fixer.scope import (  # pylint: disable=import-outside-toplevel
        inspect_single_unit_scope,
    )

    col_sliced_unit: dict[str, Any] = {
        "file": "mod.py",
        "name": "target",
        "kind": kind,
        "start": start_line,
        "end": end_line,
        "start_col": start_col,
        "end_col": end_col,
        "source_lines": sliced_lines,
        "source_lines_is_sliced": True,
        "source_lines_is_col_sliced": True,
        "allow_snippet_only": True,
    }
    if source_text is not None:
        col_sliced_unit["source_text"] = source_text
    scope_pre_sliced = inspect_single_unit_scope(col_sliced_unit)
    assert scope_pre_sliced["is_control_flow_safe"] is expect_safe, scope_pre_sliced
    assert scope_pre_sliced["inputs"] == expected_inputs
    assert scope_pre_sliced["stores"] == expected_stores

    # Control: when source_lines_is_col_sliced=False on unclipped full lines, column slicing runs once
    if source_text is not None:
        full_line_unit: dict[str, Any] = {
            "file": "mod.py",
            "name": "target",
            "kind": kind,
            "start": start_line,
            "end": end_line,
            "start_col": start_col,
            "end_col": end_col,
            "source_lines": full_lines,
            "source_lines_is_sliced": True,
            "source_lines_is_col_sliced": False,
            "source_text": source_text,
            "allow_snippet_only": True,
        }
        scope_full_line = inspect_single_unit_scope(full_line_unit)
        assert scope_full_line["is_control_flow_safe"] is expect_safe, scope_full_line
        assert scope_full_line["inputs"] == expected_inputs
        assert scope_full_line["stores"] == expected_stores


@pytest.mark.parametrize(
    ("unit", "expected_snippet"),
    [
        pytest.param(
            {
                "file": "mod.py",
                "name": "target",
                "start": 1,
                "end": 2,
                "source_lines": [
                    "def from_lines():\n",
                    "    return 1\n",
                    "def other():\n",
                    "    return 2\n",
                ],
                "file_source": "def stale_file_source():\n    return 99\n",
            },
            "def from_lines():",
            id="non_sliced_py_source_lines_beats_file_source",
        ),
        pytest.param(
            {
                "file": "mod.py",
                "name": "target",
                "start": 1,
                "end": 2,
                "source_lines": ["def explicit_unsliced():\n", "    return 3\n"],
                "source_lines_is_sliced": False,
                "source_text": "def stale_source_text():\n    return 88\n",
            },
            "def explicit_unsliced():",
            id="explicit_unsliced_py_source_lines_beats_source_text",
        ),
        pytest.param(
            {
                "file": "mod.py",
                "name": "target",
                "start": 1,
                "end": 2,
                "source_lines": ["def pre_sliced_excerpt():\n", "    return 4\n"],
                "source_lines_is_sliced": True,
                "file_source": "def full_file_source():\n    return 77\n",
            },
            "def full_file_source():",
            id="control_pre_sliced_source_lines_falls_through_to_file_source",
        ),
        pytest.param(
            {
                "file": "nb.ipynb#bad",
                "name": "target",
                "start": 1,
                "end": 2,
                "source_lines": ["def bad_nb():\n", "    return 0\n"],
            },
            None,
            id="control_malformed_notebook_anchor_returns_none",
        ),
    ],
)
def test_resolve_cached_unit_source_and_tree_matches_non_sliced_source_lines_precedence(
    unit: dict[str, Any],
    expected_snippet: str | None,
) -> None:
    """Verifies reporters.resolve_cached_unit_source_and_tree prioritizes non-sliced source_lines uniformly."""
    from pydoppelgangerhunt.reporters import (  # pylint: disable=import-outside-toplevel
        resolve_cached_unit_source_and_tree,
    )

    src, tree = resolve_cached_unit_source_and_tree(unit)
    if expected_snippet is None:
        assert src is None
        assert tree is None
    else:
        assert src is not None and expected_snippet in src
        assert isinstance(tree, ast.Module)


@pytest.mark.parametrize(
    (
        "source_lines",
        "source_lines_is_sliced",
        "source_text",
        "expect_origin_name",
        "expected_unit_snippet",
        "expected_full_snippet",
    ),
    [
        pytest.param(
            [
                '{"cells": [{"cell_type": "code", "source": '
                '["def from_lines():\\n", "    return 1\\n"]}], '
                '"metadata": {}, "nbformat": 4, "nbformat_minor": 5}\n'
            ],
            None,
            (
                '{"cells": [{"cell_type": "code", "source": '
                '["def from_text():\\n", "    return 2\\n"]}], '
                '"metadata": {}, "nbformat": 4, "nbformat_minor": 5}'
            ),
            "FULL_BUFFER",
            "def from_lines():",
            "def from_lines():",
            id="both_source_lines_and_source_text_nbformat_json_prefers_source_lines",
        ),
        pytest.param(
            ["def plain_cell():\n", "    return 3\n"],
            False,
            (
                '{"cells": [{"cell_type": "code", "source": '
                '["def from_text():\\n", "    return 2\\n"]}], '
                '"metadata": {}, "nbformat": 4, "nbformat_minor": 5}'
            ),
            "FULL_BUFFER",
            "def plain_cell():",
            "def plain_cell():",
            id="plain_python_source_lines_beats_nbformat_json_source_text",
        ),
        pytest.param(
            ["    return 3\n"],
            True,
            (
                '{"cells": [{"cell_type": "code", "source": '
                '["def from_text():\\n", "    return 3\\n"]}], '
                '"metadata": {}, "nbformat": 4, "nbformat_minor": 5}'
            ),
            "PRE_SLICED",
            "return 3",
            "def from_text():",
            id="control_pre_sliced_source_lines_falls_through_to_nbformat_source_text_for_full_text",
        ),
        pytest.param(
            [
                '{"cells": [{"cell_type": "markdown", "source": ["# md\\n"]}], '
                '"metadata": {}, "nbformat": 4, "nbformat_minor": 5}\n'
            ],
            None,
            (
                '{"cells": [{"cell_type": "code", "source": '
                '["def from_text():\\n", "    return 2\\n"]}], '
                '"metadata": {}, "nbformat": 4, "nbformat_minor": 5}'
            ),
            "PLACEHOLDER",
            None,
            None,
            id="adversarial_invalid_source_lines_nbformat_fails_closed_ahead_of_valid_source_text",
        ),
    ],
)
def test_notebook_in_memory_source_lines_beats_source_text_when_both_carry_nbformat_json(
    source_lines: list[str],
    source_lines_is_sliced: bool | None,
    source_text: str,
    expect_origin_name: str,
    expected_unit_snippet: str | None,
    expected_full_snippet: str | None,
) -> None:
    """Verifies resolve_in_memory_notebook_unit and all callers agree on source_lines > source_text precedence."""
    from pydoppelgangerhunt.fixer.source import (  # pylint: disable=import-outside-toplevel
        _load_unit_file_text,
    )
    from pydoppelgangerhunt.reporters import (  # pylint: disable=import-outside-toplevel
        resolve_cached_unit_source_and_tree,
    )
    from pydoppelgangerhunt.source_lines import (  # pylint: disable=import-outside-toplevel
        SourceLinesOrigin,
        extract_unit_source_lines_with_state,
        resolve_in_memory_notebook_unit,
    )

    unit: dict[str, Any] = {
        "file": "nb.ipynb#cell_1",
        "name": "target",
        "start": 1,
        "end": 2,
        "source_lines": source_lines,
        "source_text": source_text,
    }
    if source_lines_is_sliced is not None:
        unit["source_lines_is_sliced"] = source_lines_is_sliced

    nb_state = resolve_in_memory_notebook_unit(unit)
    lines, origin = extract_unit_source_lines_with_state(unit)
    loaded = _load_unit_file_text(unit)
    rep_src, rep_tree = resolve_cached_unit_source_and_tree(unit)

    assert origin is SourceLinesOrigin[expect_origin_name]
    if expected_unit_snippet is None:
        assert nb_state.is_invalid is True
        assert lines == []
        assert loaded is None
        assert rep_src is None and rep_tree is None
    else:
        assert nb_state.is_invalid is False
        joined_lines = "".join(lines)
        assert expected_unit_snippet in joined_lines
        assert '"nbformat"' not in joined_lines
        assert loaded is not None and expected_full_snippet is not None
        assert expected_full_snippet in loaded
        assert '"nbformat"' not in loaded
        assert rep_src is not None and expected_full_snippet in rep_src
        assert isinstance(rep_tree, ast.Module)


@pytest.mark.parametrize(
    ("source_code", "unit_kind", "expect_v_definite"),
    [
        pytest.param(
            (
                "def run(compute):\n"
                "    class Mgr:\n"
                "        def __enter__(self):\n"
                "            return self\n"
                "        def __exit__(self, *a):\n"
                "            return False\n"
                "    Mgr.__exit__ = lambda self, *a: True\n"
                "    with Mgr():\n"
                "        v = compute()\n"
                "    return v\n"
            ),
            "function",
            {"strict": False, "pedantic": False},
            id="snippet_only_direct_exit_reassign_invalidates_definite_store",
        ),
        pytest.param(
            (
                "def run(compute):\n"
                "    class Mgr:\n"
                "        def __enter__(self):\n"
                "            return self\n"
                "        def __exit__(self, *a):\n"
                "            return False\n"
                "    setattr(Mgr, '__exit__', lambda self, *a: True)\n"
                "    with Mgr():\n"
                "        v = compute()\n"
                "    return v\n"
            ),
            "function",
            {"strict": False, "pedantic": False},
            id="snippet_only_setattr_exit_invalidates_definite_store",
        ),
        pytest.param(
            (
                "class Mgr:\n"
                "    def __enter__(self):\n"
                "        return self\n"
                "    def __exit__(self, *a):\n"
                "        return False\n"
                "Mgr.__exit__ = lambda self, *a: True\n"
                "with Mgr():\n"
                "    v = compute()\n"
                "return v\n"
            ),
            "basic_block",
            {"strict": False, "pedantic": False},
            id="snippet_only_synthetic_wrapper_block_detects_exit_patch",
        ),
        pytest.param(
            (
                "def run(compute, eval=False):\n"
                "    class SafeMgr:\n"
                "        def __enter__(self):\n"
                "            return self\n"
                "        def __exit__(self, *a):\n"
                "            return False\n"
                "    flag = eval\n"
                "    with SafeMgr():\n"
                "        v = compute(flag)\n"
                "    return v\n"
            ),
            "function",
            {"strict": True, "pedantic": True},
            id="control_param_shadowing_eval_keeps_snippet_only_class_proof",
        ),
    ],
)
def test_snippet_only_scope_inspection_detects_exit_protocol_patches(
    source_code: str,
    unit_kind: str,
    expect_v_definite: dict[str, bool],
) -> None:
    """Verifies standalone source_code-only scope inspection scans exit-protocol patches when mod_tree is absent."""
    from pydoppelgangerhunt.fixer import (  # pylint: disable=import-outside-toplevel
        clear_dataflow_cache,
        inspect_single_unit_scope,
    )

    got: dict[str, bool] = {}
    for mode in ("strict", "pedantic"):
        clear_dataflow_cache()
        sc = inspect_single_unit_scope(
            {
                "name": "run",
                "file": "",
                "start": 1,
                "end": len(source_code.splitlines()),
                "kind": unit_kind,
                "source_code": source_code,
                "allow_snippet_only": True,
                "closure_strictness": mode,
            }
        )
        got[mode] = "v" in sc["definite_stores"]
    assert got == expect_v_definite


@pytest.mark.parametrize(
    ("lines", "unit_kind", "s_col", "e_col", "expect_changed"),
    [
        pytest.param(["x = 1\x00\n"], "complex_expr", 0, None, False, id="nul_byte_returns_unchanged"),
        pytest.param(
            ["(" * 300 + "1" + ")" * 300 + "\n"],
            "complex_expr",
            0,
            None,
            False,
            id="pathological_nesting_returns_unchanged",
        ),
        pytest.param(
            ["res = [x + 1 for x in items]\n"],
            "comprehension",
            6,
            None,
            True,
            id="control_valid_comprehension_infers_col_bounds",
        ),
    ],
)
def test_infer_missing_col_from_line_ast_handles_nul_and_pathological_nesting(
    lines: list[str],
    unit_kind: str,
    s_col: int | None,
    e_col: int | None,
    expect_changed: bool,
) -> None:
    """Verifies _infer_missing_col_from_line_ast uses safe_parse_ast and never raises on NUL or deep nesting."""
    from pydoppelgangerhunt.fixer.scope import (  # pylint: disable=import-outside-toplevel
        _infer_missing_col_from_line_ast,
    )

    out_s, out_e = _infer_missing_col_from_line_ast(
        lines, s_col, e_col, unit_kind=unit_kind, allow_expr_fallback=True
    )
    if expect_changed:
        assert out_s == 6 and out_e == 28
    else:
        assert (out_s, out_e) == (s_col, e_col)


def test_tree_sha256_digest_handles_lone_surrogates() -> None:
    """Verifies _get_tree_sha256_digest encodes lone surrogates without raising UnicodeEncodeError."""
    from pydoppelgangerhunt.fixer._df_cache import (  # pylint: disable=import-outside-toplevel
        _register_tree_source_lines,
        clear_dataflow_cache,
    )
    from pydoppelgangerhunt.fixer.scope import (  # pylint: disable=import-outside-toplevel
        _get_tree_sha256_digest,
    )

    clear_dataflow_cache()
    tree_cached = ast.parse("x = 'ok'\n")
    _register_tree_source_lines(tree_cached, source_text="x = '\ud800'\n")
    d_cached = _get_tree_sha256_digest(tree_cached)
    assert len(d_cached) == 64

    tree_ast = ast.Module(
        body=[ast.Expr(value=ast.Constant(value="\ud800"))],
        type_ignores=[],
    )
    d_ast = _get_tree_sha256_digest(tree_ast)
    assert len(d_ast) == 64


@pytest.mark.parametrize(
    ("start", "end", "expect_severed_closure"),
    [
        pytest.param(2, 3, True, id="kindless_placeholder_inside_enclosing_fn_records_severed_closure"),
        pytest.param(1, 4, False, id="control_kindless_placeholder_whole_fn_omits_severed_closure"),
    ],
)
def test_kindless_placeholder_unit_with_file_tree_reports_severed_closure_cell(
    start: int,
    end: int,
    expect_severed_closure: bool,
) -> None:
    """Verifies _inspect_unit_scope_impl passes file_tree to _is_subroutine_or_nested_expr_unit on placeholders."""
    from pydoppelgangerhunt.fixer import (  # pylint: disable=import-outside-toplevel
        inspect_single_unit_scope,
    )

    file_tree = ast.parse(
        "def outer():\n"
        "    x = 1\n"
        "    y = x + 2\n"
        "    return y\n"
    )
    sc = inspect_single_unit_scope(
        {
            "name": "block",
            "file": "nonexistent_placeholder.py",
            "start": start,
            "end": end,
        },
        tree=file_tree,
    )
    assert sc["is_control_flow_safe"] is False
    assert "unavailable_source_placeholder" in sc["control_flow_hazards"]
    assert ("severed_closure_cell" in sc["control_flow_hazards"]) is expect_severed_closure


def test_module_import_scans_skip_nested_scopes_and_support_match() -> None:
    """Verifies module import scanners exclude imports nested inside def/class and honor match blocks."""
    import sys  # pylint: disable=import-outside-toplevel

    nested_src = (
        "import os\n"
        "if __name__ == '__main__':\n"
        "    def _runner():\n"
        "        import json\n"
        "        return json\n"
        "    class _Box:\n"
        "        import math\n"
        "\n"
        "def target():\n"
        "    return os\n"
    )
    imported = _get_module_imported_names(nested_src, include_conditional=True)
    assert "os" in imported
    assert "json" not in imported
    assert "math" not in imported
    ins_idx = _find_module_helper_insertion_index(nested_src.splitlines(keepends=True))
    assert ins_idx == 1

    if sys.version_info >= (3, 10):
        match_src = (
            "import sys\n"
            "match sys.platform:\n"
            "    case 'win32':\n"
            "        import msvcrt\n"
            "    case _:\n"
            "        def _lazy():\n"
            "            import fcntl\n"
            "            return fcntl\n"
            "\n"
            "def worker():\n"
            "    pass\n"
        )
        cond_names = _get_module_imported_names(match_src, include_conditional=True)
        uncond_names = _get_module_imported_names(match_src, include_conditional=False)
        assert "msvcrt" in cond_names
        assert "fcntl" not in cond_names
        assert "msvcrt" not in uncond_names
        match_ins = _find_module_helper_insertion_index(match_src.splitlines(keepends=True))
        assert match_ins == 8


def test_analysis_session_tree_source_lines_enforces_lru_eviction() -> None:
    """Verifies AnalysisSession.tree_source_lines is LRU-bounded and drops entries evicted with parent_maps."""
    from pydoppelgangerhunt.fixer._df_cache import (  # pylint: disable=import-outside-toplevel
        AnalysisSession,
    )

    sess = AnalysisSession(max_bucket_entries=16)
    trees = [ast.parse(f"x = {idx}\n") for idx in range(20)]
    for idx, tree in enumerate(trees[:16]):
        sess.register_tree_source_lines(tree, (f"x = {idx}",))
        sess.get_or_build_parent_map(tree)
    # Touch trees[0] so it becomes most recently used and survives subsequent evictions
    assert sess.get_tree_source_lines(trees[0]) == ("x = 0",)
    for idx, tree in enumerate(trees[16:], start=16):
        sess.register_tree_source_lines(tree, (f"x = {idx}",))
        sess.get_or_build_parent_map(tree)

    assert len(sess.tree_source_lines) <= 16
    assert len(sess.parent_maps) <= 16
    assert trees[1] not in sess.tree_source_lines
    assert trees[0] in sess.tree_source_lines
    assert trees[-1] in sess.tree_source_lines
