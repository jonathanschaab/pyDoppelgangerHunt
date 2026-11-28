"""Shared helpers and fixtures for context-manager provenance test suites."""

from __future__ import annotations

import ast
from collections.abc import Set as AbstractSet
from typing import Any

from pydoppelgangerhunt.fixer import (
    clear_dataflow_cache,
    inspect_single_unit_scope,
)


def _late_bound_capture_safety_by_mode(
    src: str,
    unit_src: str,
    *,
    non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
) -> dict[str, bool]:
    """Returns {mode: is_control_flow_safe} for a single-line lambda unit capturing a with-bound name."""
    unit_line = next(
        idx for idx, ln in enumerate(src.splitlines(), start=1) if ln == unit_src
    )
    tree = ast.parse(src)
    results: dict[str, bool] = {}
    for mode in ("strict", "pedantic"):
        clear_dataflow_cache()
        unit_dict: dict[str, Any] = {
            "start": unit_line,
            "end": unit_line,
            "kind": "basic_block",
            "source_text": src,
            "source_code": unit_src + "\n",
            "closure_strictness": mode,
        }
        if non_suppressing_ctx_managers:
            unit_dict["non_suppressing_ctx_managers"] = set(
                non_suppressing_ctx_managers
            )
        sc = inspect_single_unit_scope(unit_dict, tree=tree)
        results[mode] = bool(sc["is_control_flow_safe"])
    return results

_PROVEN_EXIT_CLASS = (
    "class Mgr:\n"
    "    def __enter__(self):\n"
    "        return self\n"
    "    def __exit__(self, *exc_info):\n"
    "        return False\n"
)

_WITH_MGR_RUN = (
    "def run(compute):\n"
    "    with Mgr():\n"
    "        v = compute()\n"
    "    cb = lambda: v\n"
    "    return cb()\n"
)

_WITH_NULLCONTEXT_RUN = (
    "def run(compute):\n"
    "    with contextlib.nullcontext():\n"
    "        v = compute()\n"
    "    cb = lambda: v\n"
    "    return cb()\n"
)
