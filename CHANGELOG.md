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
  subroutine contains a yield expression in any non-expression context (e.g. `x = yield y`,
  `return (yield x)`, `f((yield))`, `if (yield):`), preserving explicit send types when present.
- **Strict and Non-Strict Coordinate Parsing**: Added `strict: bool = False` to `parse_unit_coord`.
  In strict mode (used by `is_valid_unit_coordinates` during patch generation), blank strings,
  colon-formatted strings, and floats are strictly rejected to prevent invalid coordinates from
  targeting line 1 or masking bugs. In non-strict mode, integer floats and diagnostic colon
  formats parse the leading integer.
- **Safe Unit File Resolution**: Unified `_load_unit_file_text` and `extract_unit_source_code`
  under `resolve_safe_unit_file_path`, checking unresolved parent directories below the repo root
  for symlinks before resolution to prevent traversing internal directory symlinks, enforcing root
  containment, and rejecting `.ipynb` files as raw Python text.
- **Forward Reference Quote Tracking**: Enhanced `_split_type_args` to track single and double
  quotes alongside bracket depth, preventing commas within literal strings from splitting
  type arguments.
- **Subroutine Unit Classification Helper**: Added `is_subroutine_unit` to detect compound
  blocks, sliding windows, and clause branches while excluding whole functions, closures,
  methods, and expressions.
- **Asynchronous Generator Detection**: Added `is_async_generator_with_return_value` and
  `is_async` unit metadata tagging to reject asynchronous generator subroutines with
  return values, yield assignments, try/finally, or async with blocks.
- **Context-Aware Subroutine Async Classification**: Subroutines inside `async def` only
  inherit `is_async=True` if the unit itself contains `await`/`async for`/`async with` or contains
  `yield` (PEP 525 async generator return rule). Pure synchronous code inside an `async def`
  remains a synchronous helper, allowing deduplication with synchronous callers.
- **Downstream Read Analysis Cache**: Implemented a thread-safe, bounded LRU cache for
  downstream live read collection keyed by content SHA-256 digest, unit coordinates,
  kind, name, and candidate outputs (relying on content digests for cache deduplication
  without timestamp fragmentation).

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
  `a, b` vs `b, a`), removing previously proposing patches on reordered independent outputs to
  prevent silent variable swapping or misalignment at call sites.
- **Subroutine Kind Parity Enforcement**: Unconditionally reject clone pairs between
  subroutine units (blocks, sliding windows, clause branches) and whole functions/methods
  within refactoring patch and helper synthesis.
- **Coordinate Clamping Across All Reporters, CLI, and Matcher**: Clamped start coordinates to
  minimum line 1 and end coordinates to at least start across all text, JSON, SARIF, Markdown, HTML,
  metrics, clustering, coverage, matcher (`merge_adjacent_clones`, `suppress_subclones`,
  `scan_target`), and CLI violation reports.
- **Dynamic Scope Dispatch via `inspect.signature`**: Replaced exception message substring matching
  in `dispatch_analyze_unit_variable_scope` with `inspect.signature` introspection, defaulting to
  `supports_trees = False` on failure.
- **Synthetic Wrapper Sentinel**: Switched subroutine synthetic wrappers to `__pdh_wrapper__`
  and guarded against flattening user functions named `_wrapper`.
- **Helper Return Type Inference**: Standardized `_infer_helper_return_type` signature to use
  explicit `unit_kind` and `is_subroutine` flags, ensuring accurate `-> Any` return type
  annotations for comprehension and complex expression units, and returning `None` for
  subroutines without outputs or return values.
- **Binding Module Structure**: Reordered imports to follow local package precedence and registered
  all public re-exports in `__all__`.
- **Default Missing/Zero Unit End Slicing**: `extract_unit_source_code` now defaults missing or
  zero `end` coordinates to `start` (slicing a single line) via `_unit_line_bounds` rather than
  slicing through end-of-file, ensuring consistent coordinate bounds handling across reporting
  and extraction pipelines.
- **Coordinate Clamping Diagnostics**: Added `DEBUG`-level logging in `matcher._unit_sloc` and
  `reporters._unit_line_bounds` whenever coordinate clamping modifies an inverted or corrupt
  coordinate.
- **Fail-Closed Unresolved End Column Downstream Detection & Dynamic Reads**: In
  `_DownstreamReadVisitor`, when `u_end_col` cannot be resolved, same-line trailing nodes on `u_end`
  are treated as downstream reads while assignments on `u_end` do not kill candidate outputs.
  Furthermore, downstream dynamic calls to `locals()`, `vars()`, `eval()`, `exec()`, or `dir()`
  are treated as reading all candidate outputs fail-closed.
- **Fail-Closed Definiteness in Clone Generator Subroutine Outputs**: Updated
  `GeneratorCloneSideData` and `resolve_clone_generator_subroutine_outputs` to treat
  `definite=None` as `set()` (fail-closed), rejecting extractions when outputs are needed without
  proven definite assignment.
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
