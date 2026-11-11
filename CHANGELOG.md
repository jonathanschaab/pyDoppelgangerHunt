# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.0.0] - 2026-11-11

### Added
- **Closure Strictness Modes (`--closure-strictness {strict,pedantic,lenient}`)**:
  Added CLI flag and configuration setting (`closure_strictness` in `pyproject.toml` and `UnitSchema`) across text/`--suggest`, `--html`, and `--patch` (full behavioral specification in `README.md`):
  - `strict` (default for reporting and preview patches with `replace_clones=False`): conservative closure-cell analysis plus a token-boundary name-and-provenance heuristic for unknown `with`/`async with` context managers (fail-open for neutrally named custom suppressors; logs a deduplicated `INFO` diagnostic and session `INFO` summary, escalated to `WARNING` under `--replace-clones`).
  - `pedantic` (default when `--replace-clones` / `replace_clones=True` is active without an explicit `closure_strictness`): treats all unproven unknown context managers as potentially suppressing unless listed in `non_suppressing_ctx_managers`, and rejects unproven implicit-dunder operations inside global-reading units.
  - `lenient` (or `--skip-pre-unit-closures`): skips pre-unit closure liveness scanning for generator subroutines and relaxes unreferenced-sibling closure checks while retaining unbound-cell, external-mutation, and mutator-escape guards.
- **`non_suppressing_ctx_managers` Configuration**:
  Added `--non-suppressing-ctx-managers` CLI flag and `non_suppressing_ctx_managers` setting in `pyproject.toml` and `UnitSchema` (supporting bare names, locally defined classes, `.Name` relative imports, dotted module paths, and `*.attr` wildcard suffixes, including `frozenset` and `collections.abc.Set` inputs), union-merged across global config and per-unit overrides.
- **Public Scope Verification, Source Origin, & Schema APIs (`SCOPE_SCHEMA_VERSION`, `SourceLinesOrigin`, `extract_unit_source_lines_with_state`, `inspect_single_unit_scope`, `scope_matches_closure_mode`, `ensure_unit_scope_closure_mode`, `UnitSchema`, `validate_unit_schema`)**:
  Exported `SCOPE_SCHEMA_VERSION = 2`, `SourceLinesOrigin` (whose `__bool__` raises `TypeError` so callers explicitly inspect `.is_placeholder`, `.is_pre_sliced`, or `.has_real_source`), `extract_unit_source_lines_with_state`, `inspect_single_unit_scope`, `scope_matches_closure_mode`, `ensure_unit_scope_closure_mode`, `UnitSchema` (`allow_snippet_only`, `source_lines_is_col_sliced`, `strict_with_suppression`), `validate_unit_schema`, and `heuristic_ctx_manager_session`.
- **Distinct `stale_source_code` and `invalid_unit_columns` Control-Flow Hazards**:
  Scope inspection records `"stale_source_code"` when an explicit `source_code` snippet diverges from the underlying module AST/source, and `"invalid_unit_columns"` when `start_col` / `end_col` coordinates are inverted or exceed physical UTF-8 line byte bounds (including off-by-one `end_col > line_byte_len`).
- **Granular Closure Hazard Telemetry**:
  Added `closure_hazard_reasons` (`cell_severing`, `conditional_live_in`, `global_escape`, `unresolved_span_bounds`, `invalid_lexical_declaration`, `unavailable_source_for_closure_mode`, `unavailable_source_placeholder`, `snippet_only_missing_enclosing_context`) to scope inspection results while keeping the patch skip-reason key `"severed closure cell"`.
- **Async Generator Delegation Hazard Guard**: Added `has_async_generator_delegation_hazard`
  (aliasing `is_async_generator_with_return_value`) guarding against PEP 525 delegation hazards
  (`async for ...: yield` losing `athrow()`/`aclose()` propagation, return values, yield
  assignments, or try/with cleanup blocks) across both subroutine and whole-function extractions.
- **Public Dataflow Cache Clearing (`clear_dataflow_cache`, `evict_tree_scopes`)**: Added
  `clear_dataflow_cache()` and `evict_tree_scopes(tree)` in `pydoppelgangerhunt` and
  `pydoppelgangerhunt.fixer` to reset memoized AST/dataflow caches or invalidate scope caches after
  in-place AST mutations.
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
- **BREAKING** - **Default `strict` Mode Tightened, `pedantic` Default for `--replace-clones`, & Strictness Precedence**:
  Default `strict` mode now models `with`/`async with` exception suppression and rejects impure calls, `with` blocks, suspension points (including async comprehensions), and non-local mutations inside global-reading units; automated call-site delegation (`generate_refactoring_patch(..., replace_clones=True)` and `--replace-clones`) defaults to `pedantic` when `closure_strictness` is unset. Explicit `closure_strictness` takes uniform precedence over `strict_with_suppression` and `skip_pre_unit_closures` across both CLI and Python APIs. Legacy or unrecognized `closure_strictness` strings (`"fast"`, `"skip"`, `"fail_closed"`) in config files or unit dicts log a warning and fall back to `"strict"`, whereas passing `--closure-strictness fast` on the CLI is rejected by `argparse` (use `--closure-strictness lenient` or `pedantic`).
- **Monotone Pair Strictness Resolution & `skip_pre_unit_closures` Defaults**:
  `skip_pre_unit_closures` and `closure_strictness` default to `None` across scope, pairing, synthesis, patch, and HTML report APIs so per-unit `UnitSchema` and `pyproject.toml` settings apply when not overridden, and `skip_pre_unit_closures=True` cannot weaken a unit's explicit `pedantic` policy.
- **BREAKING** - **Precomputed Scope Schema Verification (`SCOPE_SCHEMA_VERSION == 2`) & Placeholder Safety**:
  In `analyze_unit_variable_scope`, `synthesize_shared_helper_code`, and `generate_refactoring_patch`, caller-supplied `scope`, `scope1`, `scope2`, and `unit["scope"]` dictionaries missing `scope_schema_version == 2` or mismatching active closure flags emit a once-per-run `WARNING`, discard manual overrides, and recompute from backing module source (or fail closed with `"unavailable_source_for_closure_mode"` when source is unavailable). Snippet-only subroutine/expression units without backing module context fail closed unless `allow_snippet_only=True` is set, tree-only units lacking inline/file source emit `"unavailable_source_placeholder"`, and placeholder `u1` units are unconditionally rejected (while placeholder `u2` units are rejected for subroutines, `replace_clones=True`, or explicit `source_text2` and permitted only in whole-unit preview mode where `u1`'s verified AST supplies the helper).
- **CLI Diagnostic Stream Routing & Report-Boundary Error Isolation**:
  Heuristic context-manager summaries log exclusively to `stderr` / logging (`INFO` by default, `WARNING` under `--replace-clones`) rather than `stdout` or `--output` files. At the CLI (`--suggest`, `--diff`) and `generate_html_report` boundaries, unexpected per-pair exceptions log a warning with traceback and emit an inline diagnostic fallback without aborting the report, while direct calls to `synthesize_refactoring_suggestion` still propagate `ValueError` and `TypeError` defects.
- **Cached AST Immutability Contract (`evict_tree_scopes`)**:
  `ast.AST` trees passed via the new `tree=` / `tree1=` / `tree2=` parameters are cached by object identity and assumed immutable while cached (production scope fingerprinting checks a 512-node BFS prefix plus all immediate body/clause statement intervals); callers mutating an `ast.AST` in place must call `evict_tree_scopes(tree)` or `clear_dataflow_cache()`.
- **BREAKING** - **`--suggest` & `--html` Helper Synthesis Delegation**:
  `synthesize_refactoring_suggestion` (`--suggest`) and `generate_html_report` (`--html`) delegate non-`test_*` clone pairs to `synthesize_shared_helper_code` with shared per-file/per-notebook-cell source and AST caching, distinguishing safety rejections (`rejected as unsafe`) from structural incompatibilities (`skipped as structurally incompatible`).
- **PEP 227 Lexical Class-Scope Filtering & Pattern Irrefutability**:
  Enforced PEP 227 class-scope visibility rules across scope and closure analysis and treated `None` patterns as refutable in pattern-binding analysis.
- **Scoped AST Interval & Weak Oversized Parent-Map Caching**:
  Memoized AST scope intervals without stashing attributes on `ast.AST` nodes, unified ephemeral unit-AST tracking, and cached oversized module parent maps via weak node references so parent maps build at most once per oversized AST without retaining detached statements.
- **Explicit Column-Slicing Contract for Expression Units**:
  `comprehension` and `complex_expr` units undergo full closure-cell and control-flow hazard checks, relying on explicit `source_lines_is_sliced` and `source_lines_is_col_sliced` flags rather than indentation or expression-parsing heuristics and failing closed (`"invalid_unit_columns"`) when `end_col` exceeds physical UTF-8 line byte length.
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
  batches, and incorporated `tree` and `unit_inputs` into downstream cache keys.
- **BREAKING** - **Coordinate Failure Policy**: Required explicit, parseable `start` and `end`
  bounds in `compute_unit_spans` (with non-positive line coordinates clamped), while mutation APIs
  (`replace_unit_in_source`, `refactor_module_units`) strictly validate with
  `is_valid_unit_coordinates(u, strict=strict)` and raise `ValueError` on missing, zero,
  negative, or inverted coordinates rather than silently defaulting.
- **Subroutine Classification Scope**: Extended whole-unit kind classification in
  `is_subroutine_unit` with `class` and `data_table` (and `async_method`), preventing nested
  classes and data tables inside callables from being misclassified as subroutine blocks, and
  unified kindless-module heuristics during scope inspection.
- **BREAKING** - **Coordinate Validation Policy**: `is_valid_unit_coordinates` (when called with
  default `strict=False`) requires a positive `start` coordinate (`start > 0`) and rejects empty
  dictionaries `{}` as well as zero/negative/inverted coordinates (`end < start`).
- **Coordinate Parsing Function Evolution**: `parse_unit_coord` accepts composite line:column
  strings (e.g. `"12:0"` returning `12`), and returns `None` when `default=None` instead of
  raising `ValueError` or coercing to `0`.
- **PEP 604 Pipe Union Normalization**: Synthesized helper parameter and return type annotations
  normalize PEP 604 union pipes (`|`) to `typing.Union`, preserve `Annotated[T, ...]` metadata
  arguments, and convert `X | None` to `typing.Optional[X]` for Python 3.9 compatibility.
- **BREAKING** - **Subroutine Kind Pairing Enforcement**: Refactoring patch synthesis enforces
  that clone pairs must have matching subroutine kinds
  (`is_subroutine_unit(u1) == is_subroutine_unit(u2)`).
- **BREAKING** - **Strict Positional Output Alignment & Permuted Output Rejection**:
  `synthesize_shared_helper_code` and `generate_refactoring_patch` (via `resolve_clone_pair_outputs`) strictly pair subroutine outputs by first-store position and reject conflicting positional orders (permuted outputs) or unequal counts.
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
  and excluded `yield from` expressions.
- **Context-Aware Async Classification**: Pure synchronous code inside `async def` remains a
  synchronous helper, allowing deduplication with synchronous twins.
- **Fail-Closed Async Status on Missing Source**: When source text cannot be loaded, async
  status for yield-containing units is marked unknown (`None`), failing closed in helper
  synthesis and patch generation.
- **Receiver Metadata Path Restriction**: Receiver metadata disk inspection is confined
  strictly to `.py` files inside the repository root or current working directory.
- **Line-Splitting Public API**: Exposed `split_source_lines` and `count_physical_newlines`
  as the public line-splitting helpers in `source_lines`.

### Removed
- **Legacy Scope Dispatcher (`dispatch_analyze_unit_variable_scope`)**: Removed legacy
  dispatcher function in favor of direct invocation of `analyze_unit_variable_scope`.

### Fixed
- **Multi-Item `with` Header & Nested Context-Manager Provenance**:
  Items in multi-item `with`/`async with` statements following a suppressing or unknown item (or raising `as` target unpacking) mark their `as` targets and header `:=` walrus bindings as conditional rather than definite, while advancing provenance across `stmt.items` and `try`/`else`/`except`/`finally` blocks.
- **Write-Only Dead Conditional Store Pruning**:
  Write-only conditional stores are promoted to `inputs` only when live after the unit (in declared `outputs`, captured by an enclosing closure cell, or read downstream).
- **Global-Read Call Purity, Implicit-Dunder Hooks, & Wildcard Import Guards**:
  Restricted calls and implicit protocol hooks inside units reading an enclosing `global` variable (including rejecting async comprehensions in `strict` mode and unproven dunder operations in `pedantic` mode; see `README.md`), and rejected all calls when `from ... import *` is in scope.
- **Internal `SyntaxWarning` Suppression Fast Path & Serialization**:
  Added lock-free fast paths during AST and symbol-table parsing when `SyntaxWarning` (and `DeprecationWarning` on Python `< 3.12`) is already ignored by the host process or an enclosing suppression context on the same thread, falling back to a lock-serialized `warnings.catch_warnings()` context for cooperating internal callers (unsynchronized external threads mutating global `warnings.filters` concurrently remain subject to Python's global warning-filter model).
- **In-Memory Prioritized Unit Source Loading and Excerpt Slicing**: Prioritized in-memory unit
  source buffers (`source_text`, `file_source`, and non-sliced `source_lines`) before falling
  back to disk in `fixer/source.py`, and ensured line-1 excerpts are identified as sliced excerpts
  rather than full-file text.
- **End Column Key Precedence Alignment**: Aligned coordinate key lookup in `fixer/dataflow.py`
  to `("end_col", "end_col_offset")`, matching `start_col` and `fixer/source.py`.
- **Duplicate Skip Comment Prevention on Same-File Plans**: Prevented redundant skip comments in
  `fixer/patch.py` when clone pairs share the same file plan.
- **Defensive Type and Attribute Handling in Line Bound Resolution**: Added non-dict type guards
  and `AttributeError` handling in `resolve_unit_line_bounds` (`source_lines.py`), ensuring
  malformed unit inputs default safely to `(1, 1)`.
- **Renamed Input and Output Propagation in Clone Delegation**: Forwarded resolved secondary
  clone outputs across same-file and host-module delegations in `patch.py`, preventing clone 2
  from synthesizing delegation statements with clone 1's variable bindings.
- **Inclusive Column Boundary for End-of-Unit Closures**: Classified same-line closures beginning
  at the exclusive end column offset (`col_offset >= end_col`) as post-unit rather than intra-unit.
- **Enclosing With Block Downstream Read Analysis**: Evaluated enclosing `with` and `async with`
  block safety fail-closed against reads occurring downstream of the compound block where
  exception-handling context managers resume.
- **Single-Line Unit Whole-Line Containment Fallback**: Fell back to whole-line containment in
  `dataflow.py` when the end column offset cannot be resolved, ensuring single-line units without
  precomputed inputs discover live-ins and loop-carried dependencies.
- **Sound Loop-Carried Kill Analysis**: Snapshotted pre-unit killed bindings (with branch merging
  across `if`/`try`/`with` blocks) upon first reaching the unit during downstream read analysis,
  preventing conditional resets prior to the unit from unsoundly pruning loop-carried accumulator
  dependencies.
- **Parse Cache Key Content Isolation**: Keyed the AST parse cache on `(content_hash, len(source_text))`
  and dropped unit-dict digest fallbacks, preventing AST cache collisions from unit-level snippets.
- **Indented Method Helper Compilation**: Replaced `textwrap.dedent` during helper compilation
  validation (`patch.py`, `synthesis.py`, and `assert_valid_python`) by wrapping indented method
  definitions in `if 1:\n`, preventing spurious `IndentationError` when helpers contain column-0
  triple-quoted strings.
- **AST-Based Subroutine Async Detection**: Replaced raw source text regex scanning in `binding.py`
  with AST node traversal within unit bounds, and recognized async comprehensions at arbitrary
  nesting depths while honoring `unit["is_async"]`.
- **Pragma Refactoring Suppression**: Moved `# pragma: no-refactor` from harvesting suppression
  to refactoring patch generation (`patch.py`), keeping units visible in clone detection and
  reporting while skipping them during automated patch synthesis with a diagnostic comment.
- **Eager Generator Expression Slot Validation**: Validated exact argument positions and arities
  for built-in consumers (`min`, `max`, `dict`, `sum`, `list`, `set`, `tuple`, `frozenset`, `any`,
  `all`, `sorted`, and `str.join`) with no starred arguments, preventing non-iterated keyword or
  start arguments from being misclassified as eagerly consumed.
- **Enclosing With/Try Loop Re-entry Detection**: Evaluated loop-carried live-ins across enclosing
  compound block spans so accumulator reads that re-enter `with`/`try` blocks on subsequent loop
  iterations are detected.
- **Compound Statement Branch State Isolation in Unit Live-Ins**: Maintained branch store isolation
  across `for`, `while`, `try`, `with`, and `match` blocks during unit live-in extraction even when
  the compound statement header begins before the unit, while guarding `ExceptHandler.name` and
  `match` pattern bindings by unit bounds.
- **Sub-Line Slicing & Unmapped Statement Validation in `complex_expr` Units**: Guarded whole-line
  target pruning against sub-line units and failed closed on multi-statement semicolon lines and
  unmapped statement kinds (`ast.Assert`, bare `ast.AnnAssign`) in whole-line `complex_expr` units.
- **Cached AST Propagation & Closure Traversal Optimization**: Reused parsed ASTs across `patch.py`,
  `synthesis.py`, and `binding.py`, pre-checked `# pragma: no-refactor` on unit slices before
  full-file tokenization, deduplicated `ClassDef` body traversal, and isolated PEP 695 `TypeAlias`
  and `ClassDef` type parameters as well as `match`/`case` pattern bindings during closure analysis.
- **Whole-Line Comprehension Assignment Target & Indentation Preservation**: Unified whole-line
  `comprehension` and `complex_expr` delegation in `patch.py` so whole-line comprehension units
  retain their assignment target, leading indentation, trailing newline, and boundary pragmas.
- **Innermost Scope Selection & Subroutine Coinciding Bounds**: Tie-broke same-span inner definitions
  by column offset and excluded nested `def`/`class` nodes coinciding with subroutine bounds so
  nested generators do not pollute outer subroutine async/yield metadata.
- **Single-Line Enclosing Loop Retention for Sub-Line Units**: Retained single-line loops as
  enclosing loops when sub-line column offsets are present (`start_col` after the loop column),
  enabling loop-carried read detection on single-line loops without misclassifying whole-loop units.
- **Tree Resolution Alignment**: Unified effective AST resolution across downstream read analysis
  and closure collection in `resolve_clone_generator_subroutine_outputs`, preventing module-level
  shadowing mismatches.
- **Multi-line Import Header Support in Helper Validation**: Split synthesized helper import headers
  on triple newlines (`\n\n\n`) with CR/LF normalization during compilation validation, supporting
  multiline parenthesized imports and column-0 strings.
- **In-Memory Receiver Metadata Population**: Forwarded `source_text` and `tree` through
  `synthesize_shared_helper_code` so virtual units without disk files receive accurate async and
  receiver metadata.
- **Strict Coordinate Parameter in Module Replacements**: Added `strict: bool = True` to
  `validate_module_unit_replacements` (`patch.py`), forwarding to `refactor_module_units`.
- **Missing `end` Single-Line Fallback**: Documented `extract_unit_source_code` returning a single
  line when `end` coordinate is omitted.
- **Downstream Cache Key Unit Inputs Normalization**: Normalized `unit["inputs"]` before building
  downstream read cache keys so populated vs. unpopulated unit inputs do not collide, and guarded
  against malformed or unhashable input elements in `collect_downstream_read_names`.
- **Re-entrant Synchronization for Downstream Caching**: Used a re-entrant lock for dataflow cache
  synchronization to guarantee deadlock-free nested cache lookups.
- **Alternative Column Offset Validation**: Extended `is_valid_unit_coordinates` in `source.py`
  to validate both canonical (`start_col`, `end_col`) and offset alias keys (`start_col_offset`,
  `end_col_offset`), rejecting negative or inverted column bounds across all aliases.
- **Sliced Excerpt Guard in File Loading**: Rejected unflagged partial file fragments during unit
  file text loading when line counts are shorter than `end` or `start > 1`.
- **Strict Assignment Operator Regex in Synthesis**: Prevented equality and relational comparisons
  (`==`, `!=`, `<=`, `>=`) from matching assignment target stripping in `synthesis.py`.
- **Word-Boundary Async and Yield Classification**: Used word-boundary matching for `yield`,
  `yield from`, `await`, `async for`, and `async with` in `binding.py` to avoid false positives on
  identifiers such as `await_timeout` or `yield_step`.
- **Column-Aware Semicolon Statement Boundaries**: Extended downstream read checks in `dataflow.py`
  with column offset bounds to detect same-line downstream reads in compound or semicolon-separated
  statements.
- **Lambda Body Scope Traversal**: Handled expression-bodied scopes (`ast.Lambda`) cleanly during
  local-name extraction without leaking inner comprehension targets or nested lambda bindings.
- **Normalized Path Cross-Checking in Patch Plans**: Canonicalized file paths via
  `normalize_path_string` when rendering file patch plans in `patch.py`.
- **Scope Shadowing for Eager Built-ins**: Checked whether eager built-in consumers (`sum`, `min`,
  `max`, `any`, etc.) are shadowed by parameters or local bindings in the enclosing scope.
- **Multi-Type Union & Quoted Forward Reference Normalization**: Normalized multi-type PEP 604
  unions containing `None` (`int | str | None` -> `Optional[Union[int, str]]`), unions of quoted
  forward references (`'"NodeA" | "NodeB"'` -> `'Union["NodeA", "NodeB"]'`), and qualified
  `typing.Optional` / `typing_extensions.Optional` prefixes in `synthesis.py`.
- **Parser Unit Async Status (`is_async`)**: Populated `is_async` during AST harvesting for units
  containing asynchronous constructs, including top-level async comprehensions and sub-line column
  bounds while isolating nested synchronous scopes (`ClassDef` and `Lambda`).
- **Bracket-Stack Type Parsing**: Validated balanced bracket pairs when splitting delimited type
  annotation strings in `synthesis.py`.
- **Priority Score on Inverted Coordinates**: Restored `compute_priority_score` to return `0.0`
  when unit coordinates are inverted or non-positive.
- **Symlinked and Aliased Root Safe Path Resolution**: Permitted parent directories whose canonical
  paths match or are ancestors of `target_root` in `resolve_safe_unit_file_path`.
- **Type Annotation, Bare Target, and PEP 695 `TypeAlias` Live-In Analysis**: Visited
  `AnnAssign.annotation`, bare subscript/attribute targets (`arr[idx]: int`), and PEP 695
  `TypeAlias` statements (including type parameter bounds and defaults) while isolating lexical
  `type_params`.
- **Whole-Line Subscript, Attribute, and `AugAssign` LHS Load Pruning, RHS Operand Ordering & Arity Guard**:
  Pruned unreferenced LHS subscript/attribute loads, `AugAssign` targets, and receiver attributes
  across `Assign`, `AnnAssign`, `AugAssign`, `Return`, and `Expr` whole-line expression units,
  ordered surviving inputs by first free-load appearance on the RHS, mapped call-site arguments
  and helper parameter metadata positionally, preserved `method_binding="method"` on same-class
  whole-line expression units, and failed closed when pruned RHS input arities differ.
- **Sync vs. Async Iterator Return Annotation Cross-Check**: Preserved synchronous vs. asynchronous
  iterator return annotations (`Iterator`, `Generator`, `Iterable`, `AsyncIterator`, `AsyncGenerator`,
  including `typing.` / `collections.abc.` prefixes) according to helper async state in
  `synthesis.py`.
- **Expression Unit Kind Parity & `Return` Target Stripping**: Aligned `comprehension`/`complex_expr`
  kinds across kind-less twins while rejecting conflicting explicit kinds, unwrapped `ast.Return`
  alongside assignment statements, and cleared outputs on pruned expression scopes.
- **Downstream `For`/`With`/`Match` Target `Load` Traversal & Intra-Unit `Delete` / Walrus Scoping**:
  Recorded `Load` sub-expressions inside `for`, `with`, comprehension, and `match` targets, tracked
  `del` targets as live-ins while clearing prior stores, isolated branch-local `NamedExpr` (`:=`)
  bindings across comprehensions, `BoolOp`, and `IfExp`, and tightened sub-expression column end
  resolution on multi-statement lines.
- **PEP 695 Generic Function & Class `type_params` Handling**: Preserved top-level generic function
  `type_params` onto synthesized helper headers in `synthesis.py` (pruning unreferenced non-common
  type parameters), evaluated decorators and parameter defaults outside `type_params` scope,
  prevented recursive `TypeAlias` names from leaking into outer loads, and failed closed with
  `type_param_closure_capture` when hoisting local or class-scoped type parameter bounds to module
  scope.
- **Multiline String & PEP 701 F-String Continuation Preservation**: Preserved continuation lines
  inside multiline string and f-string literals (including multi-line `{...}` interpolations and
  lone-`\r` buffers) without zeroing `dedent_source_text` block margins or injecting indentation
  prefixes during helper synthesis.

## [1.0.0] - 2026-09-20
### Added
- Initial release of pyDoppelgangerHunt: AST-based code clone detection, redundancy gates,
  and refactoring patch generation.
