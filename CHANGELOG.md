# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Closure Strictness Modes (`--closure-strictness {strict,pedantic,lenient,fail_closed,fast,skip}`)**:
  User-facing CLI flag and configuration setting (`closure_strictness` in `pyproject.toml` and `UnitSchema`) to
  select closure strictness and context-manager suppression policy during subroutine refactoring:
  - `strict` (default): heuristic context-manager safety mode with conservative closure-cell analysis. Enforces full pre-unit and sibling closure-cell hazard analysis while using a name-and-provenance heuristic for unknown `with` / `async with` context managers (treating unknown managers as non-suppressing unless their terminal identifier/callee attribute matches one of the 34 case-insensitive suppression-idiom substrings `suppress`, `raises`, `ignor`, `subtest`, `exitstack`, `enter_context`, `enter_async_context`, `cancelscope`, `move_on_after`, `fail_after`, `catch`, `swallow`, `silen`, `quiet`, `toler`, `best_effort`, `absorb`, `trap`, `fallback`, `noop`, `no_op`, `shield`, `retry`, `attempt`, `skip`, `mute`, `muting`, `maybe`, `hush`, `discard`, `forgiv`, `squelch`, `quench`, `mask`, or their root receiver/class is proven exception-suppressing via local/function/class `__exit__` provenance; intermediate receiver-chain attribute names such as `self.retry_budget.acquire()` are not scanned; fail-open for neutrally named custom suppressors such as `Transaction()`, `Guard()`, `SpanContext()`, `SessionScope()`, or `BatchContext()`). In global-reading units, rejects attribute/subscript mutations, deletions, non-static class-body attributes (`__set_name__` descriptors), and external or `@=` augmented assignments while permitting dunder-invoking inspection/conversion built-ins, implicit boolean truthiness, read-only AST protocol syntax, and local primitive-initialized arithmetic/bitwise accumulator augmented assignments (`total = 0; total += x`, under the heuristic that operands do not mutate module-level globals inside protocol dunders). Logs a deduplicated per-manager `INFO` diagnostic with qualified manager names, a session-scoped `WARNING` summary to `stderr`/logging (`heuristic_ctx_manager_session()`), and a per-hunk advisory comment in `--patch` unified diffs when unknown managers are accepted by heuristic.
  - `pedantic` (deprecated alias: `fail_closed`): stricter static conservative mode that treats all unproven unknown `with` / `async with` context managers as potentially exception-suppressing unless allowlisted via `non_suppressing_ctx_managers`, and inside global-reading units rejects dunder-invoking built-in calls, all augmented assignments, and unproven implicit-dunder AST operations (binary/unary operators, non-identity comparisons and `in`/`not in` membership tests, attribute/`@property` and subscript reads, `for`/comprehension iteration, sequence/mapping unpacking, f-strings, `set`/`dict` key hashing, `match` class/value/mapping/sequence patterns on unproven subjects, `import`, non-builtin `raise`, and boolean truth-value evaluations) unless operands are statically proven built-in primitives or immutable literals (subject to Python's dynamic runtime boundaries such as frame introspection, metaclasses, and runtime monkeypatching).
  - `lenient` (deprecated aliases: `fast`, `skip`; or `--skip-pre-unit-closures`): skips pre-unit closure liveness scanning for generator subroutines and relaxes speculative unreferenced-sibling closure checks while keeping checks for unbound captured cells, externally mutated captured cells, intra-unit `nonlocal`/walrus mutator escapes, and sibling closures explicitly referenced via unit inputs.
  Emits a diagnostic warning when `skip_pre_unit_closures` conflicts with `closure_strictness` or when deprecated aliases (`fail_closed`, `fast`, `skip`) are used.
- **`non_suppressing_ctx_managers` Configuration**:
  Added `--non-suppressing-ctx-managers` CLI flag and `non_suppressing_ctx_managers` setting in `pyproject.toml` and `UnitSchema` (supporting bare names, locally defined classes, `.Name` relative imports, dotted module paths, and `*.attr` wildcard suffixes). Global `non_suppressing_ctx_managers` and per-unit `unit["non_suppressing_ctx_managers"]` are union-merged consistently across patch synthesis, output pairing, scope inspection, helper synthesis, and `--suggest` reporting.
- **Public Scope Mode Verification, Origin State, & Schema Helpers (`SCOPE_SCHEMA_VERSION`, `SourceLinesOrigin`, `extract_unit_source_lines_with_state`, `inspect_single_unit_scope`, `scope_matches_closure_mode`, `ensure_unit_scope_closure_mode`, `UnitSchema`, `validate_unit_schema`)**:
  Exported `SCOPE_SCHEMA_VERSION = 2`, `SourceLinesOrigin` (with explicit `.is_placeholder`, `.is_pre_sliced`, and `.has_real_source` properties and no implicit `__bool__`), `extract_unit_source_lines_with_state`, `inspect_single_unit_scope`, `scope_matches_closure_mode`, `ensure_unit_scope_closure_mode`, `UnitSchema` (including `allow_snippet_only` and `source_lines_is_col_sliced`), and `validate_unit_schema` from `pydoppelgangerhunt` and `pydoppelgangerhunt.fixer`. Also added `strict_with_suppression` as an explicit boolean configuration and per-unit `UnitSchema` option when `closure_strictness` is omitted, and `heuristic_ctx_manager_session()` for thread-safe session-scoped heuristic summary collection.
- **Distinct `stale_source_code` Control-Flow Hazard**:
  When a unit dictionary provides an explicit `source_code` snippet that diverges from the underlying file/AST at the unit's coordinates, scope inspection records `"stale_source_code"` (surfaced as `"stale source code"` in patch skip reasons) instead of conflating it with `"pathological_ast"`. Comment-stripped and whitespace-normalized `source_code` snippets that are AST- or token-equivalent to the full backing source slice are accepted without false-positive rejection.
- **Granular Closure Hazard Telemetry**:
  Added `closure_hazard_reasons` (`cell_severing`, `conditional_live_in`, `global_escape`, `unresolved_span_bounds`, `invalid_lexical_declaration`, `unavailable_source_for_closure_mode`, `unavailable_source_placeholder`, `snippet_only_missing_enclosing_context`) to scope inspection results, while keeping the patch skip-reason aggregation key stable as `"severed closure cell"` and logging granular sub-reasons at `DEBUG`.
- **Async Generator Delegation Hazard Guard**: Added `has_async_generator_delegation_hazard`
  (aliasing `is_async_generator_with_return_value`) guarding against PEP 525 delegation hazards
  (`async for ...: yield` losing `athrow()`/`aclose()` propagation, return values, yield
  assignments, or try/with cleanup blocks) across both subroutine and whole-function extractions.
- **Public Dataflow Cache Clearing (`clear_dataflow_cache`, `evict_tree_scopes`)**: Added
  `clear_dataflow_cache()` and `evict_tree_scopes(tree)` in `pydoppelgangerhunt` and
  `pydoppelgangerhunt.fixer`, allowing long-running processes, IDE servers, and batch runners to
  reset AST and downstream read memoization caches or invalidate scope caches after in-place AST
  mutations.
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
- **BREAKING** - **Default `strict` Mode Tightened & `fail_closed` Alias Remapped to `pedantic`**:
  Default `strict` mode now models `with` / `async with` exception suppression and rejects global-reading units containing arbitrary calls, `with` blocks, suspension points, attribute/subscript/descriptor mutations, deletions, or non-primitive/`@=` augmented assignments, so default `strict` rejects more candidate pairs than prior releases (neither `strict` nor `pedantic` reproduces the old `with`-unaware behavior; pass `--closure-strictness lenient` or configure `non_suppressing_ctx_managers` to migrate). In addition, `closure_strictness="fail_closed"` now remaps to `"pedantic"` (treating all unproven unknown context managers and unproven implicit-dunder operations in global-reading units as unsafe) rather than `"strict"`; legacy aliases `"fail_closed"`, `"fast"`, and `"skip"` emit a deprecation warning explaining both `"pedantic"` and `"strict"` / `"lenient"`.
- **BREAKING** - **`skip_pre_unit_closures` Default Parameter (`None`) & Monotone Closure Precedence**:
  In `resolve_clone_pair_outputs`, `resolve_clone_generator_subroutine_outputs`, `synthesize_shared_helper_code`, `generate_refactoring_patch`, `inspect_single_unit_scope`, and `analyze_unit_variable_scope` (as well as CLI invocation when `--closure-strictness` is omitted), `skip_pre_unit_closures` and `closure_strictness` default to `None` so per-unit `UnitSchema` (`closure_strictness`, `strict_with_suppression`, `skip_pre_unit_closures`) and `pyproject.toml` (`strict_with_suppression`) settings take effect when not explicitly overridden, and caller `skip_pre_unit_closures=True` cannot weaken a unit's explicit `closure_strictness="pedantic"` or `strict_with_suppression=True`.
- **BREAKING** - **Precomputed Scope Schema Verification, Override Discard, & Fail-Closed Snippet/Placeholder Units**:
  `ensure_unit_scope_closure_mode`, `analyze_unit_variable_scope`, `inspect_single_unit_scope`, `resolve_clone_pair_outputs`, and `resolve_clone_generator_subroutine_outputs` now require `scope_schema_version == SCOPE_SCHEMA_VERSION` (`2`) and matching `skip_pre_unit_closures`, `strict_with_suppression`, and `non_suppressing_ctx_managers` metadata on caller-supplied scope dictionaries. Any scope dictionary missing `scope_schema_version == 2` or computed under different closure settings is discarded (dropping manual caller overrides on `scope` / `scope1` / `scope2`) and recomputed from backing module source, or fails closed (`is_control_flow_safe=False`, `"unavailable_source_for_closure_mode"`) when backing source is unavailable. Snippet-only subroutine/expression units lacking a backing file, `tree`, or `source_text` now fail closed with `"snippet_only_missing_enclosing_context"` unless `allow_snippet_only=True` is explicitly set on the unit dictionary, and placeholder-source units of all kinds (including functions and methods) fail closed with `"unavailable_source_placeholder"`.
- **BREAKING** - **CLI Report Stream Cleanliness**:
  Heuristic context-manager advisory summaries are emitted exclusively to `stderr` / logging (`INFO` per distinct qualified manager name and a single session-scoped `WARNING` summary) rather than being printed to `stdout` or appended to `--output` text report files, keeping CLI stdout and `--output` artifacts cleanly machine-parseable.
- **BREAKING** - **Cached AST Immutability Contract & Scope Cache Eviction (`evict_tree_scopes`)**:
  `ast.AST` trees passed via `tree=` to scope and dataflow APIs are cached by object identity for scope-interval and parent-map lookups and are assumed immutable while cached (internal scope fingerprinting is an `O(1)` diagnostic heuristic over the scope header and first/last body statements, not a recursive proof of subtree immutability); callers that mutate an `ast.AST` in place (including interior statements or expressions) must call `evict_tree_scopes(tree)` (or `clear_dataflow_cache()`) after mutation to invalidate cached scope entries, and must not mutate an AST concurrently while another thread is analyzing that same AST without synchronization.
- **BREAKING** - **`--suggest` Helper Synthesis Delegation & Safety vs. Structural Diagnostics**:
  `synthesize_refactoring_suggestion` (`--suggest`) now delegates non-`test_*` clone pairs to `synthesize_shared_helper_code` using single-pass scope and pair-output resolution (while `test_*` function pairs retain `@pytest.mark.parametrize` templates), returning full synthesized helper definitions when valid or explicit rejection diagnostics that distinguish safety hazards (`rejected as unsafe`) from pair-level structural or output incompatibilities (`skipped as structurally incompatible`). Parse, OS, and recursion errors during synthesis fall back to a diagnostic comment, while unexpected runtime defects propagate normally.
- **PEP 227 Lexical Class-Scope Filtering & Pattern Irrefutability**:
  Enforced PEP 227 class-scope visibility rules across scope and closure analysis so enclosing `ClassDef` attributes are excluded from nested function, lambda, and comprehension lexical closures unless analyzing a non-function block directly inside a `ClassDef`, comprehension bodies and secondary generators inside class definitions are treated as separate inner scopes rather than class-body loads, and `None` patterns are treated as refutable in pattern-binding analysis.
- **Scoped AST Interval & Oversized Parent-Map Caching**:
  Memoized AST scope intervals in the tracked scope cache without stashing ad hoc attributes on `ast.AST` nodes, unified ephemeral unit-AST tracking across weakrefable and custom `ast.AST` nodes, and bounded parent-map construction to at most once per oversized module AST.
- **Expression-Unit Closure & Control-Flow Hazard Enforcement**:
  `comprehension` and `complex_expr` units undergo full closure-cell severing and control-flow hazard checks during scope inspection, pair output resolution, and shared helper synthesis, while preserving LHS-pruned scope state across `pedantic` synthesis and separating column-sliced excerpt flags (`source_lines_is_col_sliced`) from line-sliced excerpt flags (`source_lines_is_sliced`).
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
- **Multi-Item `with` Header & Nested Context-Manager Provenance**:
  In multi-item `with` / `async with` statements, items following a suppressing or unknown item (or following a raising `as` target unpacking) treat both their `as` targets and any header `:=` walrus bindings as conditional rather than definitely assigned. `with`-body provenance splitting advances provenance across `stmt.items` (`as` targets and `:=` walrus expressions) and across non-raising prefix statements, while recognizing imported and aliased `from contextlib import suppress` / `nullcontext` bindings and preserving positive `try` prefix and normal completion provenance across `else:`, `except:`, `finally:`, and post-`try` fallthrough.
- **Write-Only Dead Conditional Store Pruning**:
  In scope inspection, write-only conditional stores (variables conditionally assigned without unconditional assignment that are not read inside the unit and do not mutate an enclosing `nonlocal`/`global` target) are only promoted to `inputs` / `read_before_write` when the variable is live after the unit (in declared `outputs`, captured by an enclosing closure cell, or read downstream).
- **Global-Read Call Purity, Implicit-Dunder Protocol Hooks, & Wildcard Import Guards**:
  Restricted calls allowed inside units that read an enclosing `global` variable to explicit C-level built-ins (`callable`, `id`, `ord`, permitted in both `strict` and `pedantic` modes) and dunder-invoking inspection/conversion built-ins (`bin`, `bool`, `chr`, `hex`, `int`, `isinstance`, `issubclass`, `len`, `oct`, `range`, permitted in `strict` mode and rejected in `pedantic` mode), rejected context-manager blocks (`with` / `async with`), suspension points (`await`, `yield`, `yield from`, `async for`), non-static class-body attributes (`__set_name__` descriptors), `@=` augmented assignments, and deletions (`del`) inside global-reading units, rejected all augmented assignments and unproven implicit-dunder AST operations in `pedantic` mode unless operands are proven built-in primitives or literals, and rejected all calls when `from ... import *` is present in the unit, module, or enclosing function/class scopes.
- **Internal `SyntaxWarning` Suppression Fast Path & Serialization**:
  Added lock-free fast paths during AST and symbol-table parsing when `SyntaxWarning` is already ignored by the host process or an enclosing suppression context on the same thread, falling back to a lock-serialized `warnings.catch_warnings()` context for cooperating internal callers (unsynchronized external threads mutating global `warnings.filters` concurrently remain subject to Python's global warning-filter model).
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
- **Sub-Line Slicing & Unmapped Statement Validation in `complex_expr` Units**: In `patch.py` and
  `synthesis.py`, guard whole-line target pruning against sub-line units via `_is_subline_unit` and
  token line slicing, and fail closed on multi-statement semicolon lines and unmapped statement kinds
  (`ast.Assert`, bare `ast.AnnAssign`) in whole-line `complex_expr` units.
- **Cached AST Propagation & Closure Traversal Optimization**: Forwarded `f1_plan.parsed_tree` and
  `f2_plan.parsed_tree` to `find_enclosing_class` and `find_enclosing_function` in `patch.py`, reused
  `_get_or_parse_tree` in `synthesis.py` and `binding.py`, pre-checked `_NO_REFACTOR_RE` on
  pre-materialized text in `_unit_has_no_refactor_pragma`, deduplicated `ClassDef` body traversal
  in `_collect_scope_closures_uncached`, eliminated duplicate subtree walks in
  `_has_wildcard_star_import`, isolated PEP 695 `TypeAlias` and `ClassDef` type parameters as well
  as `match`/`case` pattern bindings in `_collect_scope_closures_uncached` (`_df_closures.py`), and
  recorded enclosing async context on function/class headers in `_ScopeHierarchyVisitor` (`parser.py`).
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
  in `_is_eagerly_consumed_genexp` (`_df_closures.py`) with no starred arguments (`ast.Starred`),
  `len(parent.args) == 1` for single-iterable consumers (`min`, `max`, `dict`, `list`, `set`,
  `tuple`, `frozenset`, `any`, `all`, `sorted`), and `1 <= len(parent.args) <= 2` for `sum`.
- **Priority Score on Inverted Coordinates**: Restored `compute_priority_score` to return `0.0`
  when unit coordinates are inverted or non-positive.
- **Symlinked and Aliased Root Safe Path Resolution**: Updated `resolve_safe_unit_file_path`
  to permit parent directories whose canonical paths match or are ancestors of `target_root`,
  supporting symlinked system directories.
- **Type Annotation, Bare Target, and PEP 695 `TypeAlias` Live-In Analysis**: Updated
  `_UnitLiveInVisitor` (`_df_visitors.py`) to visit `AnnAssign.annotation`, bare subscript/attribute
  targets (`arr[idx]: int`), and PEP 695 `TypeAlias` statements (including type parameter bounds
  and defaults via `_extract_type_alias_free_reads`) while isolating lexical `type_params`.
- **Whole-Line Subscript, Attribute, and `AugAssign` LHS Load Pruning, RHS Operand Ordering & Arity Guard**:
  Added `_extract_unreferenced_lhs_loads`, `_order_inputs_by_rhs_loads`, `_prune_unreferenced_lhs_scope`, and
  `_prune_pair_lhs_and_augassign_scopes` (`synthesis.py`, `_df_visitors.py`, `patch.py`) to prune LHS-only
  `ast.Load` identifiers, unreferenced `AugAssign` targets, and receiver attributes (`arr[i] = ...`,
  `self.cache[k] = ...`, `self.res1 = [x * self.factor ...]`, `res: MyType = ...`, `total += ...`),
  order surviving inputs by their first free-load appearance on the RHS (`_OuterExprVisitor.ordered_names`,
  excluding shadowed bound homonyms and appending `nonlocal` store-only names) across `Assign`, `AnnAssign`,
  `AugAssign`, `Return`, and `Expr` whole-line expression units,
  recompute receiver binding flags (`has_instance_binding`, `has_class_binding`,
  `has_receiver_access`, `binding_kind`) respecting custom `@classmethod` receiver names, map
  call-site arguments and helper parameter metadata positionally (`positional_target_map=True`,
  `dict(zip(inputs, target_inputs))`) in `_format_call_arguments` and `_format_helper_parameters`
  even when expression clones share the same variable names in swapped RHS positions,
  and fail closed when pruned RHS input arities differ (`len(scope1["inputs"]) != len(scope2["inputs"])`).
- **Sync vs. Async Iterator Return Annotation Cross-Check & Qualified Container Prefix Stripping**:
  Guarded early-return iterator prefixes in `_infer_helper_return_type` (`synthesis.py`) with `is_async`
  and `_strip_container_module_prefix(resolved_ret)` so synchronous `Iterator[...]`/`Generator[...]`/`Iterable[...]`
  annotations (including `typing.` / `collections.abc.` qualified prefixes) are preserved and not emitted
  for `async def` generators (and vice versa).
- **Safe `unit["inputs"]` Coercion in `collect_downstream_read_names`**: Coerced `unit["inputs"]` via
  `_coerce_string_frozenset` in `collect_downstream_read_names` (`_df_hazards.py`) before constructing
  the set or cache key so malformed or unhashable elements do not raise `TypeError`.
- **Unit-Slice Pre-Check & Inline Pragma Fallback Detection**: Updated `_unit_has_no_refactor_pragma`
  (`patch.py`) to pre-check contiguous leading `#` comment lines and unit lines with
  `_NO_REFACTOR_RE.search(line)` before tokenizing `full_text`, avoiding repeated full-file
  tokenization when the pragma only appears elsewhere in the file.
- **Sorted AST Scope Intervals Cache**: Sorted `intervals` by `(span, -start, -col_offset)` before
  caching on `tree._scope_intervals` in `_get_tree_scope_intervals` (`source.py`).
- **Enclosing Scope & Generic Class PEP 695 `type_params` Isolation**: Updated
  `_extract_single_node_local_names` and `_collect_scope_closures_uncached` (`_df_closures.py`) to
  record `scope_node.type_params` in `shadowed_names` and subtract enclosing `ClassDef.type_params`
  from closures inside class bases, keywords, `type_params`, and bodies while preserving free reads
  in class decorators.
- **Lenient Closure Log Level Alignment**: Downgraded the internal `skip_pre_unit_closures=True`
  log in `collect_downstream_read_names` (`_df_hazards.py`) from `logger.warning` to `logger.debug`
  to avoid duplicate/elevated warning messages alongside the CLI `INFO` notice.
- **Nested Lambda and Comprehension Closure Free-Read Filtering**: Replaced raw `ast.walk` in
  `_record_expr_closures` (`_df_closures.py`) with `_ExprClosureCollector(_BaseScopeVisitor)` and
  filtered active comprehension targets in `_BaseScopeVisitor._visit_nested_named_scope` and
  `_visit_nested_lambda` (`_df_visitors.py`), avoiding false-positive free-variable capture on
  nested lambdas (`lambda x: (lambda: x)`) and closures inside comprehensions
  (`[lambda: x for x in items]`).
- **`async_function`, `async_method`, Top-Level Async Comprehensions, Sub-Line Async/Yield Column Bounds & Nested Sync Scope Isolation**:
  Included `"async_function"` and `"async_method"` alongside `("function", "closure", "method")` in
  `_extract_unit_body_lines` (`source.py`), `_inspect_unit_scope` (`scope.py`), and
  `_build_unit_delegation_call` (`patch.py`); promoted subroutines with `has_await=True` to
  `is_async=True` in `_populate_unit_receiver_metadata` (`binding.py`) even at module top-level while
  constraining `_unit_ast_has_async_or_yield` with sub-line column offsets (`u_start_col`, `u_end_col`
  via `_is_ast_node_within_unit_bounds`);
  inspected outer-evaluated header expressions (`decorator_list`, `args.defaults`, `args.kw_defaults`,
  `bases`, `keywords` via `iter_nested_scope_header_nodes`) before skipping nested `FunctionDef`,
  `AsyncFunctionDef`, `ClassDef`, and `Lambda` bodies in `_node_is_effectively_async` (`parser.py`),
  `_unit_ast_has_async_or_yield` (`binding.py`), and `_extract_single_node_local_names` /
  `_extract_comprehension_walrus_names` (`_df_closures.py`); and
  pushed `False` onto `async_stack` for `ClassDef` and `Lambda` bodies in `_ScopeHierarchyVisitor`
  (`parser.py`).
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
- **Expression Unit Kind Parity (`_align_pair_expr_kinds`), `Return` Target Stripping & Chained Assignment Output Clearing**:
  Propagated `comprehension`/`complex_expr` kind across kind-less twins prior to `_slice_unit_token_lines`
  while rejecting conflicting explicit kinds in both `synthesize_shared_helper_code` (`synthesis.py`) and
  `generate_refactoring_patch` (`patch.py`), unwrapped `ast.Return` alongside assignment statements in
  `_strip_stmt_target_lines`, and cleared `outputs` / `precomputed_outputs` in
  `_prune_unreferenced_lhs_scope` so chained assignments (`a = b = expr`) and mixed assignment/return
  expressions (`res = expr` vs `return expr`) pair cleanly.
- **Downstream `For`/`With`/`Match` Target `Load` Traversal, Enclosing `ClassDef` Loop-Carried Traversal, Intra-Unit `Delete` & Sub-Line/Multi-Line Live-In Order**:
  Visited `node.target` in `_DownstreamReadVisitor.visit_For`, `item.optional_vars` in
  `visit_With`, and pattern children before binding in `_record_pattern_binding` and
  `_DownstreamReadVisitor.visit_MatchAs` (`_df_visitors.py`); traversed `node.body` in
  `_DownstreamReadVisitor.visit_ClassDef` whenever `n_start <= self.u_end <= n_end and not self._is_node_after_unit(node)`
  (including in `LOOP_CARRIED` mode); added `visit_Delete` to
  `_UnitLiveInVisitor` to record `ast.Name(ctx=ast.Del)` targets as live-ins and discard them from
  `self.stored`; saved and restored `self.stored` around comprehension bodies in
  `_BaseScopeVisitor._visit_comprehension`, across chained `node.values[1:]` operands in
  `visit_BoolOp`, and across intersected `body`/`orelse` branches (`stored_body & stored_else`) in
  `visit_IfExp` while recording `NamedExpr` targets in `self.stored` so intra-comprehension and
  intra-branch walrus reads do not become spurious live-ins; checked `stmt_start < u_start`,
  `start_col` with `comprehension`/`complex_expr`/`ast.Expr` or `stmt_start != u_start or start_col != stmt_col`,
  and trailing statements starting on line `u_end` (`col_offset >= end_col`) in `_resolve_unit_ast_end_col` (`_df_cache.py`) so
  single-line and multi-line sub-expression units without `end_col` or semicolon lines spanning beyond `u_end`
  do not misresolve to full or partial statements; and removed the `_is_inside` fallback
  to `generic_visit` in `_UnitLiveInVisitor.visit_Assign`, `visit_AnnAssign`, `visit_BoolOp`,
  `visit_IfExp`, and `visit_NamedExpr` so RHS values and conditional depths are always evaluated
  accurately across multi-line and sub-line unit boundaries.
- **Same-Class Whole-Line Expression `@staticmethod`, Receiver & `method_binding="method"` Guard**:
  Guarded `@staticmethod` emission in `synthesize_shared_helper_code` (`synthesis.py`) when
  `fn1_kind` or `fn2_kind` is `"instance"`, aligned `u1_eff`/`u2_eff` expression kinds prior to
  computing `fn1_kind`/`fn2_kind` and `is_in_method` in `generate_refactoring_patch` (`patch.py`),
  factored `_is_pair_in_method` (`binding.py`) so explicit `method_binding="method"` (as well as
  `super()` / mangled name references) preserves method binding and injects `self`/`cls` on
  same-class whole-line `comprehension` and `complex_expr` units inside methods, preserved
  `unit_receiver_omit` and custom `receiver_names` when pruning and ordering `unref_lhs` directly in
  `_build_unit_delegation_call` (`patch.py`), and delegated `binding_kind` recomputation in
  `_prune_unreferenced_lhs_scope` to `_determine_binding_kind`.
- **Quoted Forward Reference PEP 604 Union Normalization**: Added `_is_single_quoted_literal` in
  `synthesis.py` so `_normalize_pipe_unions` only unwraps outer quotes when the opening quote at
  index `0` closes at the final character, converting unions of quoted forward references
  (`'"NodeA" | "NodeB"'`) into `'Union["NodeA", "NodeB"]'`.
- **PEP 695 `type_params` Header Preservation, Decorator/Default Scoping & Fail-Closed Downstream `TypeAlias` Propagation**:
  Bound PEP 695 `type_params` on `FunctionDef`, `AsyncFunctionDef`, `ClassDef`, and `TypeAlias` in
  `_ScopeVisitor` (`scope.py`) and `_extract_helper_symbols` (`patch.py`); preserved top-level
  generic function `type_params` (`scope["type_params"]`, merged via `_merge_type_params` keying on
  stripped identifier names, preserving shared `*`/`**` `TypeVarTuple`/`ParamSpec` prefixes when
  bounds/defaults conflict, and ordering non-defaulted parameters before PEP 696 defaulted
  parameters) onto the synthesized helper header (`def _shared_...[T](...)`) in
  `synthesize_shared_helper_code` (`synthesis.py`) while recording top-level `type_params`
  bound/default symbol reads into `scope["type_param_free_vars"]` without polluting runtime
  `free_vars` or `inputs`; evaluated `decorator_list` and function `defaults`/`kw_defaults` outside
  `type_params` in `_extract_nested_scope_free_reads` (`_df_visitors.py`),
  `_ScopeVisitor._process_func` (`scope.py`), and `_extract_helper_symbols` (`patch.py`); normalized
  qualified iterator prefixes (`typing.`, `typing_extensions.`, `collections.abc.`) via
  `_strip_container_module_prefix` in `_infer_outputs_return_type` and `_infer_helper_return_type`
  (`synthesis.py`); visited `node.generators[0].iter` in `_ExprClosureCollector.visit_GeneratorExp`
  (`_df_closures.py`) on non-eager generator expressions so nested closures inside the outermost
  iterator are recorded at their own column offsets; recorded `alias_name` store before visiting
  `type_params` and `value` (and subtracted `own_name` in `_extract_type_alias_free_reads`) so
  recursive `TypeAlias` definitions (`type Tree = list[Tree]`) do not leak into outer `loads` or
  downstream reads; and restricted `collect_downstream_read_names` in `_inspect_unit_scope` to
  stored/nonlocal `visitor.type_aliases` (failing closed by retaining `alias_cands` in `outputs` when
  `full_src` or `ds_reads` is `None`) while unconditionally excluding `visitor.imported_names` from
  `outputs`.
- **Multiline String & PEP 701 F-String Continuation Preservation in Dedent & Helper Synthesis**: Added
  `_find_multiline_string_continuation_lines` (using `io.StringIO(text, newline=None)` for universal
  newline alignment on lone-`\r` buffers and tracking both `STRING`/`FSTRING_MIDDLE` tokens and active
  `FSTRING_START`..`FSTRING_END` ranges covering multi-line `{...}` interpolations) and
  `_dedent_preserving_multiline_strings` (`source.py`) plus `_indent_code_lines` (`synthesis.py`) so
  continuation lines inside multiline string/f-string literals neither zero out `dedent_source_text` block
  margins nor receive injected `body_indent` prefixes during helper synthesis; and wrapped indented source
  slices in `if 1:\n` in `_extract_unit_body_lines` (`source.py`), `_inspect_unit_scope` (`scope.py`), and
  `_parse_dedented_stmts` (`synthesis.py`, `patch.py`).
- **`ExceptHandler.name` Binding in Pre-Unit Closure Discovery**: Bound `handler.name` (`except ... as err:`)
  into the handler body's `bound_names` in `_collect_scope_closures_uncached` (`_df_closures.py`).
- **Partially Overlapping Nested Scope Guard in `_unit_ast_has_async_or_yield`**: Moved the nested
  `FunctionDef`/`AsyncFunctionDef`/`ClassDef`/`Lambda` skip guard ahead of the `u_start <= lineno <= u_end`
  check in `_unit_ast_has_async_or_yield` (`binding.py`) so nested scopes starting before `u_start`
  are never traversed.
- **Comprehension Generator Subscript/Attribute Targets & Class-Body Walrus Isolation**: Filtered
  comprehension loop target bindings to `isinstance(n.ctx, ast.Store)` and visited `gen.target` in
  `_BaseScopeVisitor._visit_comprehension` (`_df_visitors.py`) and `_ScopeVisitor._visit_comprehension`
  (`scope.py`) so `ast.Load` sub-expressions (`for arr[idx] in items`) are recorded as reads; and
  routed `_visit_named_expr(..., check_comp=True)` through `_can_store_in_class_scope` so walrus
  targets inside class-body comprehensions do not bind to `class_stores`.
- **Out-of-Bounds `u_start` Guard, Multi-Line AST Decorator Lookback in `_unit_has_no_refactor_pragma` & Asymmetric Expression Unit Validation**:
  Returned `False` early when `u_start > len(eff_lines)`, clamped `start_idx = min(len(eff_lines), max(0, u_start - 1))`,
  bounded `start_idx` by the earliest AST `decorator_list` line, and stepped back over contiguous `#` and `@` lines in
  `_unit_has_no_refactor_pragma` (`patch.py`), and validated both `orig_lines1` and `orig_lines2` via
  `_is_valid_expr_unit_lines` in `synthesize_shared_helper_code` (`synthesis.py`) to reject unmapped
  statement partners (e.g. `assert` or bare `return` with `stmt.value is None`).
- **Nested Generic Closure & Class-Scoped `type_params` Capture Guard, Nested Function Annotation Scoping, Multi-Line Async Comprehension Clauses, In-Place AST `_scope_intervals` Validation & CRLF Helper Compilation Check**:
  Detected PEP 695 `type_params` bounds/defaults referencing enclosing function local bindings or innermost class-scoped bindings (plus outer generic class `type_params`) hoisted outside the class
  (`type_param_closure_capture` / `has_class_scoped_type_params`, skipping only the unit's own innermost `FunctionDef` in `_get_tree_scope_intervals` and preserving harvested `kind="function"` same-class methods)
  in `_inspect_unit_scope` (`scope.py`), excluded `ast.ClassDef.name` from `_extract_single_node_local_names` (`_df_closures.py`), and pruned unreferenced non-common alpha-renamed `type_params` (with symmetric single-side `common_tp_idents` and fixed-point expansion across inter-dependent bounds/defaults) in `synthesize_shared_helper_code` (`synthesis.py`) so `synthesize_shared_helper_code` and `generate_refactoring_patch` (`patch.py`)
  fail closed when hoisting local/class bounds to module scope while allowing same-class method helpers, top-level self-referential class bounds, and inner classes unshadowed by outer class attributes; visited nested function parameter and return annotations inside
  `_ScopeVisitor._process_func` (`scope.py`) when `is_top` is `False`; checked `node.target` bounds for `ast.comprehension` clauses in
  `_unit_ast_has_async_or_yield` (`binding.py`); validated cached scope node set identity and node bounds via `_node_scope_bounds` in `_get_tree_scope_intervals` (`source.py`)
  against in-place `lineno`/`end_lineno` and structural AST mutations; and attempted direct and compound-wrapped compilation prior to triple-newline header/body splitting with CR/LF stripping (`body.lstrip("\r\n")`) in `_validate_helper_compilation` (`patch.py`).

## [1.0.0] - 2026-09-20
### Added
- Initial release of pyDoppelgangerHunt: AST-based code clone detection, redundancy gates,
  and refactoring patch generation.
