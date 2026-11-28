"""Unit tests for the shared clone-pair evaluation pipeline (_pair_eval.py)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest

from pydoppelgangerhunt.fixer._pair_eval import (
    check_pair_core_hazards,
    classify_pair_rejection_reasons,
    evaluate_clone_pair_scopes,
)
from pydoppelgangerhunt.fixer.patch import generate_refactoring_patch
from pydoppelgangerhunt.reporters import synthesize_refactoring_suggestion


@pytest.mark.parametrize(
    ("s1_patch", "s2_patch", "is_same_file", "allow_u2_ph", "stage", "expected_reason"),
    [
        pytest.param(
            {"control_flow_hazards": ["unavailable_source_placeholder"]},
            {},
            True,
            True,
            "terminal",
            "unavailable source placeholder",
            id="u1_placeholder_always_blocked",
        ),
        pytest.param(
            {},
            {"control_flow_hazards": ["unavailable_source_placeholder"]},
            True,
            True,
            "terminal",
            None,
            id="u2_placeholder_allowed_when_permitted",
        ),
        pytest.param(
            {"control_flow_hazards": ["stale_source_code"]},
            {},
            True,
            False,
            "terminal",
            "stale source code",
            id="stale_source_code_blocked",
        ),
        pytest.param(
            {"is_async": True},
            {"is_async": False},
            True,
            False,
            "compatibility",
            "mismatched async state",
            id="async_mismatch_blocked",
        ),
        pytest.param(
            {"globals": ["G"]},
            {"globals": ["G"]},
            False,
            False,
            "compatibility",
            "cross-file global variables",
            id="cross_file_globals_blocked",
        ),
        pytest.param(
            {"control_flow_hazards": ["severed_closure_cell"], "closure_hazard_reasons": ["late_mutation"]},
            {},
            True,
            False,
            "post_compat",
            "severed closure cell",
            id="severed_closure_cell_with_detail",
        ),
    ],
)
def test_check_pair_core_hazards_stages(
    s1_patch: Dict[str, Any],
    s2_patch: Dict[str, Any],
    is_same_file: bool,
    allow_u2_ph: bool,
    stage: str,
    expected_reason: str | None,
) -> None:
    """Verifies check_pair_core_hazards enforces shared blocking hazards across stages."""
    base: Dict[str, Any] = {
        "is_async": False,
        "has_yield": False,
        "nonlocals": [],
        "globals": [],
        "control_flow_hazards": [],
        "closure_hazard_reasons": [],
    }
    res = check_pair_core_hazards(
        {**base, **s1_patch},
        {**base, **s2_patch},
        is_same_file=is_same_file,
        allow_u2_placeholder=allow_u2_ph,
        check_stage=stage,
    )
    assert res.skip_reason == expected_reason


def test_preview_and_patch_share_pair_scope_evaluation_parity(tmp_path: Path) -> None:
    """Verifies preview suggestion synthesis and patch generation agree on closure hazard rejection."""
    mod = tmp_path / "pair_mod.py"
    src = (
        "def outer(items):\n"
        "    x = 1\n"
        "    cb = lambda: x\n"
        "    for it in items:\n"
        "        x = it + 1\n"
        "        res = cb()\n"
        "    return res\n"
        "\n"
        "def outer2(items):\n"
        "    x = 1\n"
        "    cb = lambda: x\n"
        "    for it in items:\n"
        "        x = it + 1\n"
        "        res = cb()\n"
        "    return res\n"
    )
    mod.write_text(src, encoding="utf-8")
    u1: Dict[str, Any] = {
        "file": str(mod),
        "name": "outer:loop",
        "kind": "compound_block",
        "start": 4,
        "end": 6,
    }
    u2: Dict[str, Any] = {
        "file": str(mod),
        "name": "outer2:loop",
        "kind": "compound_block",
        "start": 12,
        "end": 14,
    }
    eval_res = evaluate_clone_pair_scopes(
        dict(u1),
        dict(u2),
        default_strictness="pedantic",
        source_text1=src,
        source_text2=src,
        repo_root=str(tmp_path),
    )
    assert eval_res.policy.effective_mode == "pedantic"
    assert "severed_closure_cell" in eval_res.scope1["control_flow_hazards"]
    rej = classify_pair_rejection_reasons(
        u1,
        u2,
        eval_res.scope1,
        eval_res.scope2,
        pair_skip_reason=None,
        has_async_gen_hazard=False,
    )
    assert rej.has_safety_hazard is True
    assert any("severed closure cell" in r for r in rej.reasons)
    patch = generate_refactoring_patch([(1.0, u1, u2)], repo_root=str(tmp_path))
    suggestion = synthesize_refactoring_suggestion(
        u1, u2, repo_root=str(tmp_path), closure_strictness="pedantic"
    )
    assert patch == ""
    assert "severed closure cell" in suggestion.lower()
