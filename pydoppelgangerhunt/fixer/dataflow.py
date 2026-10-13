"""AST dataflow visitors, downstream read analysis, and output pairing.

This module serves as the primary dataflow analysis and output resolution facade,
orchestrating modular subcomponents:
- `_df_cache`: LRU caches, memory lifecycle management, and AST coordinate extractors.
- `_df_visitors`: AST scope visitors, live-in/out variable analysis, and traversal passes.
- `_df_closures`: Closure position classification and pre-unit closure capture.
- `_df_hazards`: Control flow hazard detection, try/with block hazards, and downstream reads.
- `_df_pairing`: Clone output variable pairing and generator subroutine output resolution.

Disk Fallback Scope:
    While reporting utilities in `reporters.py` allow both `.py` and `.ipynb` files
    for source inspection and visual diffing, dataflow analysis and patch generation
    in `dataflow.py` intentionally restrict disk-fallback loading to standard Python
    source files (`.py`). Refactoring patches cannot safely modify Jupyter notebooks
    via raw text diffs without corrupting JSON structures and cell offsets. Therefore,
    notebook units lacking embedded source text fail closed for generator subroutine
    extractions.
"""

from __future__ import annotations

import hashlib as _hashlib
import logging
from typing import TYPE_CHECKING

from pydoppelgangerhunt.fixer import (
    _df_cache,
    _df_closures,
    _df_hazards,
    _df_pairing,
    _df_visitors,
)
from pydoppelgangerhunt.fixer import source as _source

logger = logging.getLogger(__name__)

# Expose hashlib at module level so test mock.patch("...dataflow.hashlib.sha256") works
hashlib = _hashlib

# Public API exports
GeneratorCloneSideData = _df_pairing.GeneratorCloneSideData
clear_dataflow_cache = _df_cache.clear_dataflow_cache
collect_downstream_read_names = _df_hazards.collect_downstream_read_names
has_async_generator_delegation_hazard = (
    _df_hazards.has_async_generator_delegation_hazard
)
is_async_generator_with_return_value = (
    _df_hazards.is_async_generator_with_return_value
)
resolve_clone_generator_subroutine_outputs = (
    _df_pairing.resolve_clone_generator_subroutine_outputs
)
resolve_clone_pair_outputs = _df_pairing.resolve_clone_pair_outputs
resolve_generator_subroutine_outputs = (
    _df_pairing.resolve_generator_subroutine_outputs
)
CloneOutputResolution = _df_pairing.CloneOutputResolution

__all__ = [
    "CloneOutputResolution",
    "GeneratorCloneSideData",
    "clear_dataflow_cache",
    "collect_downstream_read_names",
    "has_async_generator_delegation_hazard",
    "is_async_generator_with_return_value",
    "resolve_clone_generator_subroutine_outputs",
    "resolve_clone_pair_outputs",
    "resolve_generator_subroutine_outputs",
]

# Internal cache and coordinate symbols (preserved for backwards-compatible test access)
_MAX_DOWNSTREAM_CACHE_SIZE = getattr(_df_cache, "_MAX_DOWNSTREAM_CACHE_SIZE")
_MAX_SCOPE_CACHE_SIZE = getattr(_df_cache, "_MAX_SCOPE_CACHE_SIZE")
_MAX_PARSED_TREE_CACHE_SIZE = getattr(_df_cache, "_MAX_PARSED_TREE_CACHE_SIZE")
_downstream_cache_lock = getattr(_df_cache, "_downstream_cache_lock")
_downstream_reads_cache = getattr(_df_cache, "_downstream_reads_cache")
_parsed_tree_cache = getattr(_df_cache, "_parsed_tree_cache")
_scope_parent_maps = getattr(_df_cache, "_scope_parent_maps")
_scope_stmts_by_end = getattr(_df_cache, "_scope_stmts_by_end")
_scope_closures_cache = getattr(_df_cache, "_scope_closures_cache")
_scope_loops_cache = getattr(_df_cache, "_scope_loops_cache")
_scope_try_with_cache = getattr(_df_cache, "_scope_try_with_cache")
_warned_lenient_closure = getattr(_df_cache, "_warned_lenient_closure")
_evict_tree_scopes = getattr(_df_cache, "_evict_tree_scopes")
_store_lru_cache_entry = getattr(_df_cache, "_store_lru_cache_entry")
_get_or_compute_scope_cached = getattr(_df_cache, "_get_or_compute_scope_cached")
_get_or_parse_tree = getattr(_df_cache, "_get_or_parse_tree")
_parse_source_tree = getattr(_df_cache, "_parse_source_tree")
_get_valid_unit_bounds = getattr(_df_cache, "_get_valid_unit_bounds")
_extract_first_unit_coord = getattr(_df_cache, "_extract_first_unit_coord")
_extract_unit_end_col = getattr(_df_cache, "_extract_unit_end_col")
_get_scope_parent_map = getattr(_df_cache, "_get_scope_parent_map")
_get_scope_stmts_by_end_lineno = getattr(_df_cache, "_get_scope_stmts_by_end_lineno")
_resolve_unit_ast_end_col = getattr(_df_cache, "_resolve_unit_ast_end_col")
_build_downstream_cache_key = getattr(_df_cache, "_build_downstream_cache_key")
_clear_downstream_reads_cache = getattr(_df_cache, "_clear_downstream_reads_cache")

# Internal visitors and AST inspection symbols
_record_import_names = getattr(_df_visitors, "_record_import_names")
_collect_func_args = getattr(_df_visitors, "_collect_func_args")
_collect_func_params = getattr(_df_visitors, "_collect_func_params")
_extract_nested_scope_free_reads = getattr(_df_visitors, "_extract_nested_scope_free_reads")
_extract_assigned_names = getattr(_df_visitors, "_extract_assigned_names")
_is_ast_node_within_unit_bounds = getattr(_df_visitors, "_is_ast_node_within_unit_bounds")
_extract_unit_live_in_names = getattr(_df_visitors, "_extract_unit_live_in_names")

# Internal closure analysis symbols
_extract_single_node_local_names = getattr(_df_closures, "_extract_single_node_local_names")
_extract_scope_local_names = getattr(_df_closures, "_extract_scope_local_names")
_is_confirmed_str_receiver = getattr(_df_closures, "_is_confirmed_str_receiver")
_is_eagerly_consumed_genexp = getattr(_df_closures, "_is_eagerly_consumed_genexp")
_collect_scope_closures = getattr(_df_closures, "_collect_scope_closures")
_normalize_closure_position = getattr(_df_closures, "_normalize_closure_position")
_classify_closure_position = getattr(_df_closures, "_classify_closure_position")
_collect_pre_unit_closures = getattr(_df_closures, "_collect_pre_unit_closures")

# Internal hazard and scope control flow symbols
_resolve_downstream_scope_node = getattr(_df_hazards, "_resolve_downstream_scope_node")
_find_enclosing_loops = getattr(_df_hazards, "_find_enclosing_loops")
_stmt_definitely_terminates = getattr(_df_hazards, "_stmt_definitely_terminates")
_block_definitely_terminates = getattr(_df_hazards, "_block_definitely_terminates")
_stmts_enclose_unit = getattr(_df_hazards, "_stmts_enclose_unit")
_nodes_load_any_name = getattr(_df_hazards, "_nodes_load_any_name")
_get_scope_try_and_with_blocks = getattr(_df_hazards, "_get_scope_try_and_with_blocks")
_execute_read_pass = getattr(_df_hazards, "_execute_read_pass")
_scope_reads_outputs_after_line = getattr(_df_hazards, "_scope_reads_outputs_after_line")
_compound_stmt_reads_outputs_after = getattr(
    _df_hazards, "_compound_stmt_reads_outputs_after"
)
_enclosing_try_reads_outputs = getattr(_df_hazards, "_enclosing_try_reads_outputs")
_merge_pre_unit_closure_reads = getattr(_df_hazards, "_merge_pre_unit_closure_reads")

# Internal pairing symbols and disk helper
_pair_clone_outputs = getattr(_df_pairing, "_pair_clone_outputs")
_filter_valid_output_pairs = getattr(_df_pairing, "_filter_valid_output_pairs")
_extract_effective_unit_outputs = getattr(_df_pairing, "_extract_effective_unit_outputs")
_load_unit_file_text = getattr(_source, "_load_unit_file_text")

if TYPE_CHECKING:
    from pydoppelgangerhunt.fixer._df_cache import (
        _DownstreamCacheKey,
    )
    from pydoppelgangerhunt.fixer._df_closures import (
        _ClosurePos,
    )
    from pydoppelgangerhunt.fixer._df_visitors import (
        _BaseScopeVisitor,
        _ClassScopeVisitor,
        _DownstreamReadVisitor,
        _FuncScopeVisitor,
        _OuterExprVisitor,
        _StrictPostLineVisitor,
        _UnitLiveInVisitor,
        _VisitorPassMode,
    )
else:
    _DownstreamCacheKey = getattr(_df_cache, "_DownstreamCacheKey")
    _ClosurePos = getattr(_df_closures, "_ClosurePos")
    _BaseScopeVisitor = getattr(_df_visitors, "_BaseScopeVisitor")
    _FuncScopeVisitor = getattr(_df_visitors, "_FuncScopeVisitor")
    _ClassScopeVisitor = getattr(_df_visitors, "_ClassScopeVisitor")
    _OuterExprVisitor = getattr(_df_visitors, "_OuterExprVisitor")
    _VisitorPassMode = getattr(_df_visitors, "_VisitorPassMode")
    _DownstreamReadVisitor = getattr(_df_visitors, "_DownstreamReadVisitor")
    _StrictPostLineVisitor = getattr(_df_visitors, "_StrictPostLineVisitor")
    _UnitLiveInVisitor = getattr(_df_visitors, "_UnitLiveInVisitor")
