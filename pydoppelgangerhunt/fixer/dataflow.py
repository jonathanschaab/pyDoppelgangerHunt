"""AST dataflow visitors, downstream read analysis, and output pairing.

This module serves as the primary public dataflow analysis and output resolution facade,
orchestrating modular subcomponents:
- `_df_cache`: Run-scoped ``AnalysisSession`` and immutable source-hash LRU caches.
- `_df_visitors`: AST scope visitors, live-in/out variable analysis, and traversal passes.
- `_df_closures`: Closure position classification and pre-unit closure capture.
- `_df_hazards`: Control flow hazard detection, try/with block hazards, and downstream reads.
- `_df_pairing`: Clone output variable pairing and generator subroutine output resolution.

Disk Fallback Scope:
    Source inspection (``inspect_single_unit_scope`` / ``_load_unit_file_text``) and
    dataflow helpers extract only the targeted Python code-cell text for anchored
    notebook units (``<notebook>.ipynb#cell_N``) while failing closed on unanchored
    ``.ipynb`` disk files, malformed anchors, or non-code/corrupt cells so raw nbformat
    JSON is never parsed as Python (``extract_unit_source_lines_with_state`` retains raw
    unanchored ``.ipynb`` reads solely for diff/reporting). Unified-diff patch generation
    in ``patch.py`` independently restricts writable file targets to ``.py`` files so
    refactoring patches never modify ``.ipynb`` JSON structures directly.
"""

from __future__ import annotations

import hashlib as _hashlib
import logging

from pydoppelgangerhunt.fixer._df_cache import (
    AnalysisSession,
    analysis_session,
    clear_dataflow_cache,
)
from pydoppelgangerhunt.fixer._df_hazards import (
    collect_downstream_read_names,
    has_async_generator_delegation_hazard,
    is_async_generator_with_return_value,
)
from pydoppelgangerhunt.fixer._df_pairing import (
    SAFETY_PAIR_SKIP_REASONS,
    CloneOutputResolution,
    GeneratorCloneSideData,
    PairSkipReason,
    allows_u2_source_placeholder,
    resolve_clone_generator_subroutine_outputs,
    resolve_clone_pair_outputs,
    resolve_generator_subroutine_outputs,
)

logger = logging.getLogger(__name__)

# Expose hashlib at module level so test mock.patch("...dataflow.hashlib.sha256") works
hashlib = _hashlib

__all__ = [
    "AnalysisSession",
    "CloneOutputResolution",
    "GeneratorCloneSideData",
    "PairSkipReason",
    "SAFETY_PAIR_SKIP_REASONS",
    "allows_u2_source_placeholder",
    "analysis_session",
    "clear_dataflow_cache",
    "collect_downstream_read_names",
    "has_async_generator_delegation_hazard",
    "is_async_generator_with_return_value",
    "resolve_clone_generator_subroutine_outputs",
    "resolve_clone_pair_outputs",
    "resolve_generator_subroutine_outputs",
]
