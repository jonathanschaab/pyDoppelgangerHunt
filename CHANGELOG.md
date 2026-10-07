# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Closure Strictness CLI Flags & Configuration**: Introduced
  `--closure-strictness {strict,lenient,fail_closed,fast,skip}` and `--skip-pre-unit-closures` CLI
  flags (along with `closure_strictness` setting in `pyproject.toml`) to control pre-unit
  closure scanning behavior during generator subroutine refactoring, with informational logging
  when lenient mode is active.
- **Abnormal Exit Loss Guard**: Rejects generator subroutine extraction when the candidate
  unit is enclosed in a `try`, `else`, or `except` block whose `finally` or `except` handlers
  read needed outputs, or where swallowing `except` handlers (e.g. `except GeneratorExit: pass`)
  or swallowing context managers (e.g. `with contextlib.suppress(...)`) fall through to downstream
  post-block reads, guarding against silent data loss on abnormal exit (`gen.close()`, `.throw()`,
  or loop-body exceptions) where `yield from` exits abruptly without assigning to call-site
  target outputs. Fails closed when source text or AST scope trees cannot be resolved to verify
  block cleanup safety.
- **Pre-Unit Generator Expression Capture**: Extended `_collect_pre_unit_closures` and
  `_extract_nested_scope_free_reads` to inspect `ast.GeneratorExp` in statements and within
  `ClassDef` bodies in addition to `def`, `class`, and `lambda`, capturing lazy variable
  evaluations fail-closed.
- **Mapping Container Type Inference**: Added support for `Mapping[...]` and `mapping[...]`
  container prefixes in `yield from` type unwrapping, inferring the iterated key type.
- **Generator Send Type Inference**: Infers `Generator[YieldT, Any, ReturnT]` when a generator
  subroutine contains a yield expression in assignment context (`x = yield y`), preserving explicit
  send types when present.
- **Stringified Float, Overflow, and NaN Coordinate Handling**: Extended `parse_unit_coord` to
  parse stringified floating-point coordinate strings (e.g. `"12.0"` or `"12.0:0"`) safely while
  strictly enforcing `float.is_integer()`, catching `OverflowError` (e.g. `inf`, `nan`), raising
  `ValueError` and failing validation safely on non-integral floats (e.g. `12.7`).
- **Safe Unit File Resolution**: Unified `_load_unit_file_text` and `extract_unit_source_code`
  under `_resolve_safe_unit_file_path`, enforcing repo root / CWD containment, symlink traversal
  prevention (rejecting symlinked directories and files within the repo while allowing symlinked
  repo roots), and rejecting `.ipynb` files as raw Python text.
- **Forward Reference Quote Tracking**: Enhanced `_split_type_args` to track single and double
  quotes alongside bracket depth, preventing commas within literal strings from splitting
  type arguments.
- **Subroutine Unit Classification Helper**: Added `is_subroutine_unit` to detect compound
  blocks, sliding windows, and clause branches while excluding whole functions, closures,
  methods, and expressions.
- **Asynchronous Generator Detection**: Added `is_async_generator_with_return_value` and
  `is_async` unit metadata tagging to reject asynchronous generator subroutines with
  return values or downstream output assignments.
- **Asynchronous Subroutine Inheritance & Context Isolation**: Subroutine units
  (compound blocks, sliding windows, clause branches) enclosed within an `async def`
  function inherit `is_async=True`, ensuring synthesized helper functions are defined as
  `async def` and invoked via `await`. This preserves coroutine context isolation and
  deliberately skips cross-context refactoring between asynchronous and synchronous callers
  to avoid mixing event-loop-bound and blocking execution models.
- **Colon-Formatted Coordinate Support**: Added support for colon-separated coordinates
  (e.g. `"line:col"` in line coordinates and `"col:line"` in column fields, extracting
  the leading coordinate prefix) in `parse_unit_coord` and `is_valid_unit_coordinates`.
- **Downstream Read Analysis Cache**: Implemented a thread-safe, bounded LRU cache for
  downstream live read collection keyed by content SHA-256 digest, unit coordinates,
  kind, name, candidate outputs, and caller-provided `mtime` or timestamp metadata
  (relying on content digests for cache invalidation without disk `stat()` overhead).

### Changed
- **Fail-Closed Escaping Closure Protection (Generator Subroutines)**: For generator
  subroutines, pre-unit closures, lambdas, generator expressions, and class methods that capture
  candidate output variables are treated as live downstream reads to prevent unbound local
  errors on generator return.
- **Fail-Closed Downstream Read Resolution**: When downstream read sets cannot be definitively
  determined (e.g. sliced lines, unparsable source, or invalid coordinate bounds), all candidate
  outputs are treated as needed, failing closed on non-definitely-assigned outputs.
- **Strict Positional Output Alignment & Unequal Arity Rejection**: Refactored `_pair_clone_outputs`
  to strictly pair subroutine outputs by first-store position and fail closed if common variables
  have conflicting positional indices or unequal output counts (rejecting permuted outputs like
  `a, b` vs `b, a`), preventing silent variable swapping or misalignment at call sites.
- **Subroutine Kind Parity Enforcement**: Unconditionally reject clone pairs between
  subroutine units (blocks, sliding windows, clause branches) and whole functions/methods
  within refactoring patch and helper synthesis.
- **Coordinate Clamping Across All Reporters, CLI, and Matcher**: Clamped start coordinates to
  minimum line 1 and end coordinates to at least start across all text, JSON, SARIF, Markdown, HTML,
  metrics, clustering, coverage, matcher (`merge_adjacent_clones`, `suppress_subclones`,
  `scan_target`), and CLI violation reports.
- **Dynamic Scope Dispatch via `inspect.signature`**: Replaced exception message substring matching
  in `dispatch_analyze_unit_variable_scope` with `inspect.signature` introspection.
- **Synthetic Wrapper Sentinel**: Switched subroutine synthetic wrappers to `__pdh_wrapper__`
  and guarded against flattening user functions named `_wrapper`.
- **Helper Return Type Inference**: Standardized `_infer_helper_return_type` signature to use
  explicit `unit_kind` and `is_subroutine` flags, ensuring accurate `-> Any` return type
  annotations for comprehension and complex expression units, and returning `None` for
  subroutines without outputs or return values.
- **Binding Module Structure**: Reordered imports to follow local package precedence and registered
  all private re-exports in `__all__`.
- **Default Missing/Zero Unit End Slicing**: `extract_unit_source_code` now defaults missing or
  zero `end` coordinates to `start` (slicing a single line) via `_unit_line_bounds` rather than
  slicing through end-of-file, ensuring consistent coordinate bounds handling across reporting
  and extraction pipelines.
- **Coordinate Clamping Diagnostics**: Added `DEBUG`-level logging in `matcher._unit_sloc` and
  `reporters._unit_line_bounds` whenever coordinate clamping modifies an inverted or corrupt
  coordinate.
- **Fail-Closed Unresolved End Column Downstream Detection**: In `_DownstreamReadVisitor`, when
  `u_end_col` cannot be resolved, same-line trailing nodes on `u_end` are treated as downstream
  reads while assignments on `u_end` do not kill candidate outputs, eliminating fail-open
  whole-line bypasses.
- **Fail-Closed Definiteness in Clone Generator Subroutine Outputs**: Updated
  `resolve_clone_generator_subroutine_outputs` to default missing definiteness information to
  empty sets for hand-built or mocked scopes lacking `definite_stores` and `inputs`, rejecting
  extractions when outputs are needed without proven definite assignment.
- **Single-Unit Scope Analysis Interface**: Introduced `inspect_single_unit_scope` taking a single
  `tree` keyword argument, eliminating positional `tree1=tree2` scope analysis naming hacks.
- **AST Preprocessing Memoization**: Memoized per-scope parent maps, end-line statement indices,
  loop ranges, closures, and try/with blocks directly on the scope AST node, eliminating
  redundant whole-tree walks across candidate pairs.
- **Cross-File Missing Plan Text Fallback**: Fixed `f2_text` in `patch.py` to fall back to `None`
  instead of `""` when `f2_plan` is missing, allowing `_load_unit_file_text` to properly load
  the second file from disk.
- **Refactoring Patch Logging Level**: Downgraded redundant lenient closure mode logging in
  `generate_refactoring_patch` from `INFO` to `DEBUG`, avoiding duplicate CLI output.

## [1.0.0] - 2026-09-20
### Added
- Initial release of pyDoppelgangerHunt: AST-based code clone detection, redundancy gates,
  and refactoring patch generation.
