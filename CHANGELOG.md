# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Closure Strictness Modes (`--closure-strictness {strict,lenient,fail_closed,fast,skip}`)**:
  User-facing CLI flag and configuration setting (`closure_strictness` in `pyproject.toml`) to
  select closure strictness during generator subroutine refactoring (`strict` / `fail_closed`
  for full soundness vs `lenient` / `fast` / `skip` to reduce false-positive pair rejections).
  Emits a diagnostic warning when `--skip-pre-unit-closures` is combined with strict mode.
- **Async Generator Delegation Hazard Guard**: Added `has_async_generator_delegation_hazard`
  (aliasing `is_async_generator_with_return_value`) guarding against PEP 525 delegation hazards
  (`async for ...: yield` losing `athrow()`/`aclose()` propagation, return values, yield
  assignments, or try/with cleanup blocks) across both subroutine and whole-function extractions.
- **Public Dataflow Cache Clearing (`clear_dataflow_cache`)**: Added `clear_dataflow_cache()`
  in `pydoppelgangerhunt.fixer` (aliasing `_clear_downstream_reads_cache`) allowing long-running
  processes, IDE servers, and batch runners to reset AST and downstream read memoization caches.
- **Closure Warning Cache Lifecycle**: Added `clear_closure_warning_cache` in `config.py`
  (invoked by `clear_dataflow_cache`) for reset in long-running processes.
- **Subroutine Unit Classification**: Added `is_subroutine_unit` to detect compound blocks,
  sliding windows, and clause branches while excluding whole functions, methods, classes, and
  data tables.
- **Pre-Unit Generator Expression Capture**: Extended closure detection to inspect
  `ast.GeneratorExp` within statements and class bodies alongside functions, classes, and lambdas.
- **Form-Feed Preserving Dedent (`dedent_source_text`)**: Added `dedent_source_text` in
  `fixer/source.py` ensuring leading form-feed characters (`\x0c`) are preserved without
  distorting relative block indentation.
- **Compiler Validation in Test Assertions (`assert_valid_python`)**: Added `assert_valid_python`
  test helper verifying that synthesized code is valid Python at both parser and bytecode
  compiler stages.

### Changed
- **Loop-Carried Liveness for Unit Inputs**: Updated `collect_downstream_read_names` and
  `resolve_clone_generator_subroutine_outputs` to preserve unit inputs that are updated in
  enclosing loops across iterations, preventing accumulator variables from resetting across loop
  batches. Forwarded `tree` and `unit_inputs` into `_build_downstream_cache_key`.
- **Coordinate Failure Policy**: Required explicit `start` and `end` line coordinates in
  `compute_unit_spans`, `replace_unit_in_source`, and `refactor_module_units`, raising
  `ValueError` on missing or invalid line coordinates rather than silently defaulting.
- **Subroutine Classification Scope**: Extended whole-unit kind classification in
  `is_subroutine_unit` with `class` and `data_table` (and `async_method`), preventing nested
  classes and data tables inside callables from being misclassified as subroutine blocks.
  Unified duplicate kindless-module heuristics from `_inspect_unit_scope`.
- **Coordinate Validation Policy**: `is_valid_unit_coordinates` (when called with default
  `strict=False`) requires a positive `start` coordinate (`start > 0`) and rejects empty
  dictionaries `{}` as well as zero/negative/inverted coordinates (`end < start`).
- **Coordinate Parsing Function Evolution**: `parse_unit_coord` accepts composite line:column
  strings (e.g. `"12:0"` returning `12`), and returns `None` when `default=None` instead of
  raising `ValueError` or coercing to `0`.
- **Keyword-Only Helper Type Inference**: `_infer_helper_return_type` signature transitioned to
  keyword-only parameters following `meta2` (`*, is_async=..., ...`).
- **PEP 604 Pipe Union Normalization**: Synthesized helper parameter and return type annotations
  normalize PEP 604 union pipes (`|`) to `typing.Union`, preserve `Annotated[T, ...]` metadata
  arguments, and convert `X | None` to `typing.Optional[X]` for Python 3.9 compatibility.
- **Subroutine Kind Pairing Enforcement**: Refactoring patch synthesis enforces that clone pairs
  must have matching subroutine kinds (`is_subroutine_unit(u1) == is_subroutine_unit(u2)`).
- **Strict Positional Output Alignment & Unequal Arity Rejection**: `_pair_clone_outputs`
  strictly pairs subroutine outputs by first-store position and rejects conflicting positional
  orders or unequal counts.
- **Downstream Read Analysis Cache Key**: Cache keys incorporate full 256-bit SHA-256 source
  content digests, unit digests, file path, line/column coordinates, unit kind and name,
  subroutine flag (`is_subroutine`), candidate outputs, unit inputs, and closure strictness flag.
- **Liveness Model Dynamic Reads**: Added `globals()` to dynamic read detection alongside
  `locals()`, `vars()`, `eval()`, `exec()`, and `dir()`.
- **Intra-Unit Closure Output Capture Rejection**: Rejects generator subroutine extraction
  when an intra-unit closure captures candidate outputs, taking into account column offsets.
- **Enclosing `with` Block Exit Safety**: Any enclosing `with` or `async with` block whose
  outputs are read downstream causes generator subroutines to fail closed.
- **Lexical Yield Assignment Scoping**: Confined `has_yield_assignment` to the outer unit scope
  (`len(self._scope_stack) <= 1`) and excluded `yield from` expressions.
- **Context-Aware Async Classification**: Pure synchronous code inside `async def` remains a
  synchronous helper, allowing deduplication with synchronous twins.
- **Fail-Closed Async Status on Missing Source**: When source text cannot be loaded, async
  status for yield-containing units is marked unknown (`None`), failing closed in helper
  synthesis and patch generation.
- **Receiver Metadata Path Restriction**: `_populate_unit_receiver_metadata` confines disk
  inspection strictly to `.py` files inside the repository root or CWD.

### Deprecated
- **Private Deprecated Helpers**: Deprecated `_compute_unit_spans` in favor of `compute_unit_spans`.

### Removed
- **Legacy Scope Dispatcher (`dispatch_analyze_unit_variable_scope`)**: Removed internal
  dispatcher function in favor of direct invocation of `analyze_unit_variable_scope`.
- **Private Line-Splitting Regex (`_PHYSICAL_LINE_RE`)**: Dropped `_PHYSICAL_LINE_RE` from
  `source_lines.__all__` in favor of public helper functions `split_source_lines` and
  `count_physical_newlines`.
- **Legacy Subroutine Name Colon Fallback**: Removed colon detection in unit names from
  `is_subroutine_unit` to avoid false subroutine classifications.

### Fixed
- **Indented Method Helper Compilation**: Replaced unconditional
  `compile(dedent_source_text(helper_code), ...)` in `patch.py` with
  `_validate_helper_compilation`, splitting top-level import headers from indented method
  definitions to prevent spurious `IndentationError`.
- **Fail-Open Fallback in File Loading**: Hardened in-memory fallbacks in `_load_unit_file_text`
  (`source.py`) with `_is_sliced_excerpt_for_file`, rejecting sliced line excerpts lacking the flag
  where `len(lines) < u_end` or `u_start > 1` so downstream analysis is not run against partial
  file fragments.
- **Strict Assignment Operator Regex in Synthesis**: Updated `m_assign` in `synthesis.py`
  with negative lookbehind and lookahead (`(?<![=<>!:])=(?!=)`) to prevent equality comparisons
  (`==`, `!=`, `<=`, `>=`) from being mistaken for assignment statements.
- **Word-Boundary Async and Yield Classification**: Replaced raw substring containment in
  `binding.py` with word-boundary regular expressions (`\b(yield|yield\s+from)\b` and
  `\b(await|async\s+for|async\s+with)\b`), eliminating false-positive async classifications on
  identifiers like `await_timeout` or `yield_step`.
- **Column-Aware Semicolon Statement Boundaries**: Extended `_StrictPostLineVisitor`,
  `_scope_reads_outputs_after_line`, and `_enclosing_try_reads_outputs` in `dataflow.py` with column
  offset checks (`u_end_col`), correctly detecting same-line downstream reads in compound or
  semicolon-separated statements.
- **Robust AST Scope Body Stack Initialization**: Safely initialized traversal stack in
  `_extract_single_node_local_names` (`dataflow.py`) for non-list scope bodies
  (such as `ast.Lambda`), preventing `TypeError` on expression body traversal.
- **Normalized Path Cross-Checking in Patch Plans**: Canonicalized file paths via
  `normalize_path_string` in `_render_file_patch_plan` (`patch.py`) when filtering retained
  candidate spans, preventing false-negative replacement drops under path syntax drift.
- **Airtight Scope Shadowing for Eager Built-ins**: Added `_extract_scope_local_names` in
  `dataflow.py` to check whether eager built-in consumers (`sum`, `min`, `max`, `any`, etc.)
  are shadowed by parameters or local variables/functions in the enclosing scope.
- **Multi-Type Union Normalization with `None`**: Updated `_format_union_or_optional` in
  `synthesis.py` so union annotations containing three or more types with `None` (e.g.
  `int | str | None`) normalize to `Optional[Union[int, str]]` for Python 3.9 compatibility.
- **Parser Emits Unit Async Status (`is_async`)**: AST parsing populates `is_async` on code
  units containing asynchronous constructs (`AsyncFunctionDef`, `await`, `async for`, `async with`).
- **Qualified `Optional` Prefix Support**: Updated `_wrap_conditional_optional` in `synthesis.py`
  to recognize `typing.Optional` and `typing_extensions.Optional`.
- **Import Alias Isolation in Local Names**: Added `continue` after import and global/nonlocal nodes
  in `_extract_single_node_local_names`, preventing aliased import names from leaking into scopes.
- **Bracket-Stack Type Parsing**: Replaced scalar bracket depth in `_split_delimited_type_string`
  with an explicit bracket stack to reject mismatched bracket pairs (e.g. `Tuple[int, str)`).
- **Call Argument Validation for Generator Expressions**: Verified that `genexp` is an argument
  in `_is_eagerly_consumed_genexp` before classifying it as consumed by eager built-ins or `.join`.
- **Priority Score on Inverted Coordinates**: Restored `compute_priority_score` to return `0.0`
  when unit coordinates are inverted or non-positive.
- **Symlinked and Aliased Root Safe Path Resolution**: Updated `resolve_safe_unit_file_path`
  to permit parent directories whose canonical paths match or are ancestors of `target_root`,
  supporting symlinked system directories.

## [1.0.0] - 2026-09-20
### Added
- Initial release of pyDoppelgangerHunt: AST-based code clone detection, redundancy gates,
  and refactoring patch generation.
