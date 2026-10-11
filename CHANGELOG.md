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
- **BREAKING** - **Coordinate Failure Policy**: Required explicit `start` and `end` line
  coordinates in `compute_unit_spans`, `replace_unit_in_source`, and `refactor_module_units`,
  validating with `is_valid_unit_coordinates(u, strict=strict)` and raising `ValueError` on
  missing, zero, negative, or inverted coordinates rather than silently defaulting.
- **Subroutine Classification Scope**: Extended whole-unit kind classification in
  `is_subroutine_unit` with `class` and `data_table` (and `async_method`), preventing nested
  classes and data tables inside callables from being misclassified as subroutine blocks.
  Unified duplicate kindless-module heuristics from `_inspect_unit_scope`.
- **BREAKING** - **Coordinate Validation Policy**: `is_valid_unit_coordinates` (when called with
  default `strict=False`) requires a positive `start` coordinate (`start > 0`) and rejects empty
  dictionaries `{}` as well as zero/negative/inverted coordinates (`end < start`).
- **Coordinate Parsing Function Evolution**: `parse_unit_coord` accepts composite line:column
  strings (e.g. `"12:0"` returning `12`), and returns `None` when `default=None` instead of
  raising `ValueError` or coercing to `0`.
- **Keyword-Only Helper Type Inference**: `_infer_helper_return_type` signature transitioned to
  keyword-only parameters following `meta2` (`*, is_async=..., ...`).
- **PEP 604 Pipe Union Normalization**: Synthesized helper parameter and return type annotations
  normalize PEP 604 union pipes (`|`) to `typing.Union`, preserve `Annotated[T, ...]` metadata
  arguments, and convert `X | None` to `typing.Optional[X]` for Python 3.9 compatibility.
- **BREAKING** - **Subroutine Kind Pairing Enforcement**: Refactoring patch synthesis enforces
  that clone pairs must have matching subroutine kinds
  (`is_subroutine_unit(u1) == is_subroutine_unit(u2)`).
- **BREAKING** - **Strict Positional Output Alignment & Permuted Output Rejection**:
  `_pair_clone_outputs` strictly pairs subroutine outputs by first-store position and rejects
  conflicting positional orders (permuted outputs) or unequal counts.
- **Downstream Read Analysis Cache Key**: Cache keys incorporate full 256-bit SHA-256 source
  content digests (hex string), line/column coordinates, candidate outputs, unit inputs, and
  closure strictness flag. By omitting file path, unit names, and kinds, cache entries are cleanly
  shared across identical files and clones.
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
- **Disk-Prioritized Unit Source Loading and Excerpt Slicing**: Prioritized disk reading in
  `_load_unit_file_text` (`fixer/source.py`) before falling back to in-memory `source_lines`,
  eliminated duplicate fallback logic, and removed the `u_start > 1` constraint in
  `_is_sliced_excerpt_for_file` so line-1 excerpts are correctly identified as sliced excerpts
  rather than full-file text.
- **End Column Key Precedence Alignment**: Aligned coordinate key lookup in `_extract_unit_end_col`
  (`fixer/dataflow.py`) to `("end_col", "end_col_offset")`, matching the precedence established in
  `start_col` and `fixer/source.py`.
- **Duplicate Skip Comment Prevention on Same-File Plans**: Guarded diagnostic attachment in
  `_record_skip_comment` (`fixer/patch.py`) with `f2_plan is not f1_plan` and safeguarded reason
  counters with `.get(reason_key, 0) + 1`, preventing redundant comments when clone pairs share
  the same file plan.
- **Module-Level Hoisting of Unit Start Helper**: Hoisted `_unit_start` to module scope in
  `fixer/patch.py`, eliminating per-iteration redefinition inside `generate_refactoring_patch`.
- **Defensive Type and Attribute Handling in Line Bound Resolution**: Added non-dict type guards
  and expanded exception catching to include `AttributeError` in `resolve_unit_line_bounds`
  (`source_lines.py`), ensuring malformed unit inputs gracefully default to `(1, 1)`.
- **Renamed Input and Output Propagation in Clone Delegation**: Forwarded resolved `target_outs2`
  (and caller outputs `c_tout`) to `_delegate_unit_in_plan` across same-file and host module
  delegations in `patch.py`, preventing clone 2 from incorrectly synthesizing delegation statements
  with clone 1's variable bindings.
- **Inclusive Column Boundary for End-of-Unit Closures**: Updated `_classify_closure_position` in
  `dataflow.py` to use `c_col >= u_end_col` rather than strict inequality, properly classifying
  same-line closures beginning at the exclusive end column offset as post-unit rather than
  intra-unit.
- **Accurate Target Statement Scoping in Enclosing With Blocks**: Added statement target detection
  (`_stmt_assigns_to_any_name`) and identity checks (`s is not target_stmt`) in
  `_enclosing_try_reads_outputs` (`dataflow.py`), preventing in-place accumulator updates at
  index >= 1 from being falsely marked as post-unit output reads when column offsets are unresolved.
- **Harness Public Cache API**: Migrated test harness fixtures in `conftest.py` from private
  `_clear_downstream_reads_cache` to public `clear_dataflow_cache`.
- **Single-Line Unit Whole-Line Containment Fallback**: Updated `_is_ast_node_within_unit_bounds`
  in `dataflow.py` to fall back to whole-line containment when `u_end_col` cannot be resolved
  (`u_end_col is None`), harmonizing containment with the fail-closed downstream fallback and
  ensuring single-line units without precomputed inputs accurately discover live-ins and
  loop-carried dependencies.
- **Sound Loop-Carried Kill Analysis**: Snapshotted `self.killed` (with proper branch merging across
  if/try/with blocks) as `unit_entry_killed` upon first reaching the unit in
  `_DownstreamReadVisitor`, preventing conditional resets prior to the unit from unsoundly
  pruning loop-carried accumulator dependencies.
- **Parse Cache Key Content Isolation**: Keyed `_parsed_tree_cache` on
  `(content_hash, len(source_text))` and dropped unit-dict digest fallbacks, preventing AST cache
  collisions from unit-level snippets or stale IDE entries.
- **Indented Method Helper Compilation**: Replaced `textwrap.dedent` in
  `_validate_helper_compilation` (`patch.py`), `synthesize_shared_helper_code` (`synthesis.py`),
  and `assert_valid_python` (`conftest.py`) by wrapping indented method definitions in `if 1:\n`,
  preventing margin zeroing and spurious `IndentationError` when helpers contain triple-quoted
  strings with lines at column 0.
- **AST-Based Subroutine Async Detection**: Replaced raw source text regex scanning in
  `_populate_unit_receiver_metadata` (`binding.py`) with AST node traversal within unit bounds,
  and extended `_visit_comprehension` and `_inspect_unit_scope` (`scope.py`) to recognize async
  comprehensions at arbitrary nesting depths and honor `unit["is_async"]`.
- **Pragma Refactoring Suppression**: Moved `# pragma: no-refactor` from harvesting suppression
  to refactoring patch generation (`patch.py`), ensuring code units with `# pragma: no-refactor`
  remain fully visible in clone detection and reporting while being skipped during automated patch
  synthesis with a diagnostic comment.
- **Loop-Carried Liveness in Enclosing Blocks**: Evaluates statement span `[start_line, stmt_end]`
  with base `_DownstreamReadVisitor` in `_scope_reads_outputs_after_line` (`dataflow.py`), correctly
  detecting loop-carried reads at the top of an enclosing loop prior to `with` / `try` blocks.
- **Tree Resolution Alignment**: Unified `effective_tree` across downstream read analysis and
  closure collection in `resolve_clone_generator_subroutine_outputs` (`dataflow.py`), preventing
  module-level shadowing mismatch.
- **Multi-line Import Header Support in Helper Validation**: Updated `_validate_helper_compilation`
  (`patch.py`) and `assert_valid_python` (`tests/helpers.py`) to split headers cleanly on `\n\n\n`
  rather than fragile line-by-line checks, supporting multiline parenthesized imports and column-0
  strings. Dropped redundant compilation re-validation in `generate_refactoring_patch`.
- **In-Memory Receiver Metadata Population**: Forwarded `source_text` and `tree` through
  `synthesize_shared_helper_code` (`synthesis.py`) to `_populate_unit_receiver_metadata` and
  `is_subroutine_unit`, ensuring virtual units without disk files accurately receive async and
  receiver metadata.
- **Transactional Dataflow Cache Clearance**: Wrapped `generate_refactoring_patch` in a
  `try: ... finally: clear_dataflow_cache()` block to prevent memoized ASTs and downstream read
  caches from lingering across patch runs.
- **Strict Coordinate Parameter in Module Replacements**: Added `strict: bool = True` to
  `validate_module_unit_replacements` (`patch.py`), forwarding to `refactor_module_units`.
- **Missing `end` Single-Line Fallback**: Documented `extract_unit_source_code` safely returning a
  single line when `end` coordinate is missing.
- **Downstream Cache Key Unit Inputs Normalization**: Normalized `effective_unit_inputs` from
  `unit.get("inputs")` before generating `_build_downstream_cache_key`, ensuring calls with
  populated vs unpopulated unit inputs produce distinct cache keys in `_downstream_reads_cache`.
- **Re-entrant Synchronization for Downstream Caching**: Migrated `_downstream_cache_lock` in
  `dataflow.py` from `threading.Lock` to `threading.RLock`, guaranteeing re-entrancy safety
  across nested multi-cache lookups without deadlocking.
- **Alternative Column Offset Validation**: Extended `is_valid_unit_coordinates` in `source.py`
  to validate both canonical (`start_col`, `end_col`) and offset alias keys (`start_col_offset`,
  `end_col_offset`), rejecting negative or inverted column bounds across all aliases.
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
