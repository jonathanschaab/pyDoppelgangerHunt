# Changelog

All notable changes to `pyDoppelgangerHunt` are documented in this file.

## [Unreleased]

### Breaking & Behavioral Changes
- **`closure_strictness` Mode Split (`strict` vs. `pedantic`)**:
  - `strict` (default, `strict_with_suppression = False`) enforces full pre-unit and sibling closure-cell hazard analysis while using a name-and-provenance heuristic for unknown `with` / `async with` context managers (treating unknown managers as non-suppressing unless their identifier matches suppression idioms such as `suppress`, `raises`, `ignore`, `subTest`, `ExitStack`, `enter_context`, or is proven exception-suppressing via local/function/class `__exit__` provenance).
  - `pedantic` (aliases: `fail_closed`, `strict_with`; or `strict_with_suppression = True`) treats all unproven unknown `with` / `async with` context managers as potentially exception-suppressing unless allowlisted via `non_suppressing_ctx_managers`. Note that `"fail_closed"` now resolves to `"pedantic"` (`strict_with_suppression = True`) rather than `"strict"`.
  - `lenient` (aliases: `fast`, `skip`; or `skip_pre_unit_closures = True`) skips pre-unit closure cell severing checks.
- **Precomputed Scope Schema Version (`SCOPE_SCHEMA_VERSION = 2`)**:
  - `scope_matches_closure_mode` and `ensure_unit_scope_closure_mode` require `scope_schema_version == 2` and explicit boolean `skip_pre_unit_closures`, `strict_with_suppression`, and normalized `non_suppressing_ctx_managers` fields on precomputed scope dictionaries. Unmarked or legacy scope dictionaries are automatically recomputed when source/AST is available and fail closed when source is unavailable.
- **`_extract_scope_local_names(..., include_classes=False)` Default**:
  - `_extract_scope_local_names` now defaults to `include_classes=False` so nested `ClassDef` body attributes do not leak into enclosing function local sets; callers requiring class-level local extraction must pass `include_classes=True`.
- **`_is_irrefutable_pattern(None)` Contract**:
  - `_is_irrefutable_pattern(None)` now returns `False` rather than `True`. Bare wildcard `case _:` patterns (`ast.MatchAs(pattern=None, name=None)`) are handled directly on the `ast.MatchAs` node before recursing into `pattern`.
- **Expression-Unit Closure & Control-Flow Hazard Enforcement**:
  - `comprehension` and `complex_expr` units now undergo full closure-cell severing and control-flow hazard checks during scope inspection, pair output resolution, and shared helper synthesis.

### Added
- **`strict_with_suppression` & `non_suppressing_ctx_managers` Configuration**:
  - Added `--non-suppressing-ctx-managers` CLI flag and `strict_with_suppression` / `non_suppressing_ctx_managers` settings in `pyproject.toml` and `UnitSchema`.
  - `strict_with_suppression` is retained as a backward-compatible boolean alias (`true` maps to `closure_strictness = "pedantic"` when `closure_strictness` is omitted; explicit `closure_strictness` takes precedence and logs a conflict warning). Users should migrate to the unified `closure_strictness = "strict" | "pedantic" | "lenient"` configuration knob.
  - Global `non_suppressing_ctx_managers` and per-unit `unit["non_suppressing_ctx_managers"]` are union-merged consistently across `patch.py`, `_df_pairing.py`, `scope.py`, and `synthesis.py`.
- **Distinct `stale_source_code` Control-Flow Hazard**:
  - When a unit dictionary provides an explicit `source_code` snippet that diverges from the underlying file/AST at the unit's coordinates, `_inspect_unit_scope` now records `"stale_source_code"` (surfaced as `"stale source code"` in patch skip reasons) instead of conflating it with `"pathological_ast"`. Comment-stripped and whitespace-normalized `source_code` snippets that are AST- or token-equivalent to the source slice are accepted without false-positive rejection.
- **Granular Closure Hazard Telemetry**:
  - Added `closure_hazard_reasons` (`cell_severing`, `loop_reassigned_capture`, `conditional_live_in`, `global_escape`, `unresolved_span_bounds`, `invalid_lexical_scope_declarations`, `unavailable_source_for_closure_mode`, `unavailable_source_placeholder`) to scope inspection results and patch skip-reason reporting.

### Fixed
- **Multi-Item `with` Header Definite Assignment**:
  - In multi-item `with` / `async with` statements, items following a suppressing or unknown item (or following a raising `as` target unpacking) now treat both their `as` targets and any header `:=` walrus bindings as conditional rather than definitely assigned across both intra-unit block analysis and enclosing-scope pre-bound cell traversal.
- **Nested Helper Promotion Purity & Wildcard Import Soundness**:
  - Removed environment-dependent and side-effectful builtins (`dir`, `vars`, `print`, `input`, `open`, `breakpoint`, `help`, `eval`, `exec`, `compile`, `__import__`) from `_PURE_GLOBAL_SAFE_BUILTINS` and rejected nested helper promotion when enclosing scopes contain `from ... import *`.
- **AST Scope Cache Key Hashing & Warning Isolation**:
  - Precomputed `_ScopeCacheKey` hashes at construction time to avoid repeated $O(n)$ `frozenset` hashing on cache lookups, and suppressed `SyntaxWarning` emissions during the `compile(...)` fast path in `_has_invalid_lexical_scope_declarations`.
