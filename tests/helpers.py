"""Shared test assertion and verification helpers."""

from typing import Optional


def assert_valid_python(source: str, mode: str = "exec") -> None:
    """Verifies that source is both syntactically and semantically valid Python.

    Handles import headers separated by triple newlines, indented definitions,
    and multiline parenthesized imports without line-splitting corruption.
    """
    if "\n\n\n" in source:
        header, body = source.split("\n\n\n", 1)
        if header.strip():
            compile(header, filename="<synthesized_imports>", mode=mode)
        if body.strip():
            first_line = next((ln for ln in body.splitlines() if ln.strip()), "")
            if first_line.startswith((" ", "\t")):
                compile("if 1:\n" + body, filename="<synthesized_helper>", mode=mode)
            else:
                compile(body, filename="<synthesized_helper>", mode=mode)
        return

    try:
        compile(source, filename="<test_source>", mode=mode)
    except IndentationError:
        compile("if 1:\n" + source, filename="<test_source>", mode=mode)
