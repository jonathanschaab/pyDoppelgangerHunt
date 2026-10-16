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
- **Scoped Subroutine Colon Fallback**: Restricted colon detection in unit names to units with
  unspecified kind (`not kind`), preventing named functions containing colons from false
  subroutine classifications while maintaining support for legacy kind-less blocks.
- **Batch Boundary Dataflow Cache Clearance**: Clarified that `clear_dataflow_cache()` is exposed
  as a public API for callers to invoke at batch boundaries rather than running automatically
  within `generate_refactoring_patch`, avoiding cache thrashing during concurrent patch runs
  while allowing long-running runners to release memoized ASTs.
- **Loop-Carried Liveness for Unit Inputs**: Updated `collect_downstream_read_names` and
  `resolve_clone_generator_subroutine_outputs` to preserve unit inputs that are updated in
  enclosing loops across iterations, preventing accumulator variables from resetting across loop
  batches. Forwarded `tree` and `unit_inputs` into `_build_downstream_cache_key`.
- **BREAKING** - **Coordinate Failure Policy**: Required explicit, parseable `start` and `end`
  bounds in `compute_unit_spans` (with non-positive line coordinates clamped), while mutation APIs
  (`replace_unit_in_source`, `refactor_module_units`) strictly validate with
  `is_valid_unit_coordinates(u, strict=strict)` and raise `ValueError` on missing, zero,
  negative, or inverted coordinates rather than silently defaulting.
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
- **Private Line-Splitting Regex Visibility (`_PHYSICAL_LINE_RE`)**: Dropped `_PHYSICAL_LINE_RE`
  from `source_lines.__all__` in favor of public helper functions `split_source_lines` and
  `count_physical_newlines`.

### Deprecated
- **Private Deprecated Helpers**: Deprecated `_compute_unit_spans` in favor of `compute_unit_spans`.

### Removed
- **Legacy Scope Dispatcher (`dispatch_analyze_unit_variable_scope`)**: Removed internal
  dispatcher function in favor of direct invocation of `analyze_unit_variable_scope`.

### Fixed
- **In-Memory Prioritized Unit Source Loading and Excerpt Slicing**: Prioritized in-memory unit
  source buffers (`source_text`, `file_source`, and non-sliced `source_lines`) before falling
  back to disk in `_load_unit_file_text` (`fixer/source.py`), eliminated duplicate fallback
  logic, and removed the `u_start > 1` constraint in `_is_sliced_excerpt_for_file` so line-1
  excerpts are correctly identified as sliced excerpts rather than full-file text.
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
- **Enclosing With Block Downstream Read Analysis**: Enclosing with and async with block safety
  is evaluated fail-closed via `_compound_stmt_reads_outputs_after` to detect reads occurring
  downstream of the compound block (where exception-handling context managers resume), superseding
  obsolete intra-block statement heuristics (`_stmt_assigns_to_any_name`).
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
  and `assert_valid_python` (`tests/helpers.py`) by wrapping indented method definitions in `if 1:\n`,
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
- **Eager Generator Expression Slot Validation**: In `_is_eagerly_consumed_genexp` (`_df_closures.py`),
  validate exact argument positions for built-in consumers (`parent.args[0]` for `min`, `max`, `dict`,
  `sum`, `list`, etc.) and `str.join`, preventing non-iterated keyword arguments (`dict(gen=...)`,
  `min(..., default=...)`) and start arguments (`sum(..., gen)`) from being misclassified as eagerly consumed.
- **Enclosing With/Try Loop Re-entry Detection**: In `_scope_reads_outputs_after_line` (`_df_hazards.py`),
  evaluate `live_ins - loop_killed` across the compound block span `[stmt_start, stmt_end]`, ensuring
  loop-carried accumulator reads that re-enter the compound block on subsequent iterations are detected.
- **Compound Statement Branch State Isolation in Unit Live-Ins**: In `_UnitLiveInVisitor` and
  `_extract_unit_live_in_names` (`_df_visitors.py`), iterate scope statements directly and maintain
  branch store isolation across `for`, `while`, `try`, `with`, and `match` blocks even when the compound
  statement header begins before `u_start`. Guard `ExceptHandler.name` binding with `_is_inside(h)` and
  record `MatchAs`/`MatchStar`/`MatchMapping` pattern bindings inside unit bounds via `visit_MatchAs`.
- **AugAssign Target Pruning Cross-Unit Collision & Sub-Line Slicing**: In `patch.py` and `synthesis.py`,
  guard AugAssign target pruning against subline units via `_is_subline_unit` and token line slicing,
  exclude each unit's own unreferenced target via `_should_prune_augassign_targets` when checking for
  RHS input collisions across clones (supporting same-name targets without desynchronizing `inputs` vs `inputs2`),
  and fail closed on multi-statement semicolon lines and unmapped statement kinds (`ast.Assert`, bare `ast.AnnAssign`)
  in whole-line `complex_expr` units.
- **Cached AST Propagation & Closure Traversal Optimization**: Forwarded `f1_plan.parsed_tree` and
  `f2_plan.parsed_tree` to `find_enclosing_class` and `find_enclosing_function` in `patch.py`, reused
  `_get_or_parse_tree` in `synthesis.py` and `binding.py`, pre-checked `_NO_REFACTOR_RE` on
  pre-materialized text in `_unit_has_no_refactor_pragma`, deduplicated `ClassDef` body traversal
  in `_collect_scope_closures_uncached`, eliminated duplicate subtree walks in
  `_has_wildcard_star_import`, and isolated PEP 695 `TypeAlias` type parameters in
  `_extract_single_node_local_names` (`_df_closures.py`).
- **Whole-Line Comprehension Assignment Target & Indentation Preservation**: Unified
  `target_unit.get("kind") in ("comprehension", "complex_expr")` in `_build_unit_delegation_call`
  (`patch.py`), ensuring whole-line comprehension units retain their assignment target (`lhs = `),
  leading indentation, trailing newline, and boundary pragmas while sub-line comprehensions continue
  to emit inline expressions.
- **Innermost Scope Selection & Subroutine Coinciding Bounds**: Delegated `_find_innermost_ast_scope`
  (`binding.py`) to `_find_innermost_enclosing_node` (`source.py`), tie-breaking same-span inner
  definitions by column offset and excluding nested `def`/`class` nodes whose line bounds coincide
  with a subroutine body (`is_subroutine=True`) so nested generators do not pollute outer subroutine
  async/yield metadata.
- **Single-Line Enclosing Loop Retention for Sub-Line Units**: Updated `_find_enclosing_loops`
  (`_df_hazards.py`) to retain single-line loops (`(l_start, l_end) == (u_start, u_end)`) when
  sub-line column offsets (`u_start_col` or `u_end_col`) are present, enabling loop-carried read
  detection on single-line loops.
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
- **Type Annotation, Bare Target, and PEP 695 `TypeAlias` Live-In Analysis**: Updated
  `_UnitLiveInVisitor` (`_df_visitors.py`) to visit `AnnAssign.annotation`, bare subscript/attribute
  targets (`arr[idx]: int`), and PEP 695 `TypeAlias` statements (including type parameter bounds
  and defaults via `_extract_type_alias_free_reads`) while isolating lexical `type_params`.
- **Whole-Line Subscript and Attribute LHS Load Pruning**: Added `_extract_unreferenced_lhs_loads`,
  `_prune_unreferenced_lhs_scope`, and `_prune_pair_lhs_and_augassign_scopes` (`synthesis.py`,
  `patch.py`) to prune LHS-only `ast.Load` identifiers and receiver attributes (`arr[i] = ...`,
  `self.cache[k] = ...`, `self.res1 = [x * self.factor ...]`, `res: MyType = ...`), recompute
  receiver binding flags (`has_instance_binding`, `has_class_binding`, `has_receiver_access`,
  `binding_kind`) respecting custom `@classmethod` receiver names (`receiver_kind == "class"`), and
  allow asymmetric `AugAssign` vs. `Assign` target pruning when RHS arities match.
- **Sync vs. Async Iterator Return Annotation Cross-Check**: Guarded early-return iterator
  prefixes in `_infer_helper_return_type` (`synthesis.py`) with `is_async` so synchronous
  `Iterator[...]`/`Generator[...]`/`Iterable[...]` annotations are not emitted for `async def`
  generators (and vice versa).
- **Inline and Multi-Line Pragma Fallback Detection**: Updated `_unit_has_no_refactor_pragma`
  fallback (`patch.py`) to scan contiguous leading `#` comment lines and inspect unit lines
  directly with `_NO_REFACTOR_RE.search(line)` so inline trailing `# pragma: no-refactor` comments
  are detected even if tokenization fails.
- **Sorted AST Scope Intervals Cache**: Sorted `intervals` by `(span, -start, -col_offset)` before
  caching on `tree._scope_intervals` in `_get_tree_scope_intervals` (`source.py`).
- **Enclosing Scope PEP 695 `type_params` Shadowing**: Updated `_extract_single_node_local_names`
  (`_df_closures.py`) to record `scope_node.type_params` names so generic function/class type
  parameters shadowing built-ins (e.g., `def compute[sum](items):`) are included in `shadowed_names`.
- **Lenient Closure Log Level Alignment**: Downgraded the internal `skip_pre_unit_closures=True`
  log in `collect_downstream_read_names` (`_df_hazards.py`) from `logger.warning` to `logger.debug`
  to avoid duplicate/elevated warning messages alongside the CLI `INFO` notice.
- **Nested Lambda and Comprehension Closure Free-Read Filtering**: Replaced raw `ast.walk` in
  `_record_expr_closures` (`_df_closures.py`) with `_ExprClosureCollector(_BaseScopeVisitor)` and
  filtered active comprehension targets in `_BaseScopeVisitor._visit_nested_named_scope` and
  `_visit_nested_lambda` (`_df_visitors.py`), avoiding false-positive free-variable capture on
  nested lambdas (`lambda x: (lambda: x)`) and closures inside comprehensions
  (`[lambda: x for x in items]`).
- **`async_function` and `async_method` Whole-Unit Extraction and Delegation**: Included
  `"async_function"` and `"async_method"` alongside `("function", "closure", "method")` in
  `_extract_unit_body_lines` (`source.py`), `_inspect_unit_scope` (`scope.py`), and
  `_build_unit_delegation_call` (`patch.py`).
- **Whole-Loop Column Offset Guard in `_find_enclosing_loops`**: Cached loop `col_offset` in
  `_scope_loops_cache` (`_df_cache.py`, `_df_hazards.py`) and only classified same-span loops
  (`(l_start, l_end) == (u_start, u_end)`) as enclosing when `u_start_col is not None and l_col < u_start_col`,
  preventing whole-loop units with column metadata from treating themselves as enclosing loops.
- **Comprehension-Aware RHS Free-Load Check (`_node_references_name`)**: Replaced flat `ast.walk`
  in `_node_references_name` (`synthesis.py`) with `_OuterExprVisitor` so bound comprehension loop
  variables on the RHS do not block LHS subscript/attribute load or `AugAssign` target pruning.
- **Lambda Root Expression Scope Isolation in `_extract_single_node_local_names`**: Guarded
  `_COMPREHENSION_TYPES` and `_INNER_SCOPE_TYPES` at the top of `while stack:` in
  `_extract_single_node_local_names` (`_df_closures.py`) so `ast.Lambda` nodes whose immediate
  body is a comprehension or nested lambda do not leak comprehension targets or inner lambda stores.
- **Expression Unit Kind Parity and Chained Assignment Output Clearing**: Propagated
  `comprehension`/`complex_expr` kind across kind-less twins while rejecting conflicting explicit
  kinds in `synthesize_shared_helper_code` (`synthesis.py`), and cleared `outputs` /
  `precomputed_outputs` in `_prune_unreferenced_lhs_scope` so chained assignments (`a = b = expr`)
  pair cleanly with single-target assignments (`c = expr`).
- **Downstream `For` and `With` Target `Load` Traversal & Multi-Line `Assign` Live-In Order**:
  Visited `node.target` in `_DownstreamReadVisitor.visit_For` and `item.optional_vars` in
  `visit_With` (`_df_visitors.py`) to capture subscript/attribute `Load` reads (`for arr[i] in ...`,
  `with cm() as obj.slot[i]:`), and removed the `_is_inside` fallback to `generic_visit` in
  `_UnitLiveInVisitor.visit_Assign` and `visit_AnnAssign` so `value` is always evaluated before
  `targets` on multi-line statements starting before `u_start`.

## [1.0.0] - 2026-09-20
### Added
- Initial release of pyDoppelgangerHunt: AST-based code clone detection, redundancy gates,
  and refactoring patch generation.
