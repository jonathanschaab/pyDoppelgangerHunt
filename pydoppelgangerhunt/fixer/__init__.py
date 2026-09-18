"""Automated refactoring engine for extracting duplicated code into shared helpers.

This package provides modules for source inspection, variable scoping, receiver binding,
shared helper synthesis, and unified diff patch generation.
"""

from __future__ import annotations

from pydoppelgangerhunt.fixer.binding import (
    _base_unit_name,
    _extract_child_indentation,
    _find_innermost_enclosing_node,
    _get_enclosing_receiver_kind,
    _has_receiver_reference,
    _inspect_enclosing_node,
    _is_method_of_class,
    _is_same_file_path,
    _normalize_file_path,
    _populate_unit_receiver_metadata,
    _prune_unshared_receivers,
    _resolve_effective_binding,
    find_enclosing_class,
    find_enclosing_function,
)
from pydoppelgangerhunt.fixer.patch import (
    _FilePatchPlan,
    _adjust_line_for_replacements,
    _build_unit_delegation_call,
    _build_whole_method_delegation,
    _derive_module_import_path,
    _has_unconditional_terminal_return,
    _module_imports_target,
    _render_file_patch_plan,
    check_units_overlap,
    filter_overlapping_clone_units,
    generate_refactoring_patch,
    refactor_module_units,
)
from pydoppelgangerhunt.fixer.scope import (
    BUILTIN_NAMES,
    _ScopeVisitor,
    _analyze_block_assignment,
    _block_terminates,
    _demote_deletions_to_conditional,
    _determine_binding_kind,
    _extract_deleted_names,
    _inspect_unit_scope,
    _is_irrefutable_case,
    _is_irrefutable_pattern,
    _is_mangled_name,
    _normalize_receiver_order,
    _rank_param_kind,
    _unfold_receiver_attribute,
    _walrus_assignment_in_expr,
    _walrus_in_sequence,
    analyze_unit_variable_scope,
)
from pydoppelgangerhunt.fixer.source import (
    _detect_indent_step,
    _extract_docstring_end_line,
    _extract_module_docstring_end_line,
    _extract_unit_body_lines,
    _find_header_cookie_boundary,
    _find_module_helper_insertion_index,
    _find_sig_colon,
    _get_module_imported_names,
    _insert_imports_into_module,
    _is_docstring_node,
    _scan_sig_line,
    _slice_unit_token_lines,
    extract_unit_comments_and_pragmas,
    replace_unit_in_source,
    slice_source_by_token_range,
)
from pydoppelgangerhunt.fixer.synthesis import (
    TYPING_SYMBOLS,
    _extract_required_typing_imports,
    _format_call_arguments,
    _format_helper_docstring,
    _format_helper_parameters,
    _infer_helper_return_type,
    _merge_types,
    synthesize_shared_helper_code,
)

__all__ = [
    "analyze_unit_variable_scope",
    "check_units_overlap",
    "extract_unit_comments_and_pragmas",
    "filter_overlapping_clone_units",
    "find_enclosing_class",
    "find_enclosing_function",
    "generate_refactoring_patch",
    "refactor_module_units",
    "replace_unit_in_source",
    "slice_source_by_token_range",
    "synthesize_shared_helper_code",
]
