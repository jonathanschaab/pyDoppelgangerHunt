# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Closure Strictness Modes (`--closure-strictness {strict,lenient}`)**: User-facing CLI flag
  and configuration setting (`closure_strictness` in `pyproject.toml`) to select closure
  strictness during generator subroutine refactoring (`strict` for full soundness vs `lenient`
  to reduce false-positive pair rejections).
- **Async Generator Delegation Hazard Guard**: Added `has_async_generator_delegation_hazard`
  (aliasing `is_async_generator_with_return_value`) guarding against PEP 525 delegation hazards
  (`async for ...: yield` losing `athrow()`/`aclose()` propagation, return values, yield
  assignments, or try/with cleanup blocks) across both subroutine and whole-function extractions.
- **Mapping Container and Generator Send Type Inference**: Added support for `Mapping[...]` and
  `mapping[...]` container prefixes in `yield from` unwrapping, and inferred
  `Generator[YieldT, Any, ReturnT]` for non-expression yield statements.
- **Pre-Unit Generator Expression Capture**: Extended closure detection to inspect
  `ast.GeneratorExp` within statements and class bodies alongside functions, classes, and lambdas.
- **Subroutine Unit Classification**: Added `is_subroutine_unit` to detect compound blocks,
  sliding windows, and clause branches while excluding whole functions and methods.

### Changed
- **PEP 604 Pipe Union Normalization**: Synthesized helper parameter and return type annotations
  normalize PEP 604 union pipes (`|`) to `typing.Union` across all parameter annotations for
  Python 3.9 compatibility.
- **Enclosing `with` Block Exit Safety**: Any enclosing `with` or `async with` block whose
  outputs are read downstream causes generator subroutines to fail closed, preventing context
  manager premature exit or cleanup hazards across arbitrary context managers (locks, streams).
- **Strict Coordinate Validation in Patch Generation**: `is_valid_unit_coordinates(strict=True)`
  requires both `start` and `end` as positive integers with `end >= start` and same-line
  `end_col >= start_col`. Patch generation fails closed on invalid units rather than degrading
  multi-line units into single lines. Permissive coordinate clamping remains reserved for
  reporting and diagnostic displays.
- **Context-Aware Async Classification**: Pure synchronous code (no `await`, `async for`,
  `async with`, or `yield`) inside `async def` remains a synchronous helper, allowing
  deduplication with synchronous twins.
- **Fail-Closed Async Status on Missing Source**: When source text cannot be loaded, async
  status for yield-containing units is marked unknown (`None`), failing closed in helper
  synthesis and patch generation.
- **Downstream Read Analysis Cache Key**: Cache keys incorporate full 256-bit SHA-256 source
  content digests, unit digests, file path, line/column coordinates, unit kind and name,
  subroutine flag (`is_subroutine`), candidate outputs, and closure strictness flag.
- **Strict Positional Output Alignment & Unequal Arity Rejection**: `_pair_clone_outputs`
  strictly pairs subroutine outputs by first-store position and rejects conflicting positional
  orders or unequal counts.
- **Liveness Model Dynamic Reads**: Added `globals()` to dynamic read detection alongside
  `locals()`, `vars()`, `eval()`, `exec()`, and `dir()`.
- **Intra-Unit Closure Output Capture Rejection**: Rejects generator subroutine extraction
  when an intra-unit closure captures candidate outputs.
- **Configuration Resolution Decoupling**: Moved `resolve_closure_strictness_mode` from
  `dataflow.py` to `config.py`, and imported dataflow symbols directly in `fixer/__init__.py`.

### Fixed
- **Priority Score on Inverted Coordinates**: Restored `compute_priority_score` to return `0.0`
  when unit coordinates are inverted or non-positive.
- **Source Lines Slicing Heuristic**: Fixed `is_sliced_unit_source_lines` by removing premature
  `len(lines) < e_d` heuristic so `extract_unit_source_code` slices appropriately.
- **Sliced Lines Guard Scope in `_load_unit_file_text`**: Ensured `source_lines_is_sliced` only
  guards the `source_lines` branch, preserving `source_text`, `file_source`, and disk fallbacks.
- **Safe Unit File Resolution Performance**: Optimized `resolve_safe_unit_file_path` to eliminate
  per-ancestor `parent.resolve()` calls and remove redundant symlink checks.
- **Cross-File Missing Plan Text Fallback**: Fixed `f2_text` in `patch.py` to fall back to `None`
  instead of `""` when `f2_plan` is missing, allowing proper loading from disk.

## [1.0.0] - 2026-09-20
### Added
- Initial release of pyDoppelgangerHunt: AST-based code clone detection, redundancy gates,
  and refactoring patch generation.
