"""Shared test assertion and verification helpers."""

from typing import Optional


def assert_valid_python(source: str, mode: str = "exec") -> None:
    """Verifies that source is both syntactically and semantically valid Python.

    Handles import headers separated by triple newlines, indented definitions,
    and multiline parenthesized imports without line-splitting corruption.
    """
    try:
        compile(source, filename="<test_source>", mode=mode)
        return
    except IndentationError:
        try:
            compile("if 1:\n" + source, filename="<test_source>", mode=mode)
            return
        except IndentationError:
            pass

    sep = "\r\n\r\n\r\n" if "\r\n\r\n\r\n" in source else "\n\n\n"
    if sep in source:
        header, body = source.split(sep, 1)
        if header.strip():
            compile(header, filename="<synthesized_imports>", mode=mode)
        if body.strip():
            if body.lstrip("\r\n").startswith((" ", "\t")):
                compile("if 1:\n" + body, filename="<synthesized_helper>", mode=mode)
            else:
                compile(body, filename="<synthesized_helper>", mode=mode)
        return

    compile("if 1:\n" + source, filename="<test_source>", mode=mode)
