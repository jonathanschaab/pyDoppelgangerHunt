# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **Subroutine Kind Parity Enforcement**: Unconditionally reject clone pairs between subroutine units (blocks, sliding windows, clause branches) and whole functions/methods within refactoring patch and helper synthesis. This prevents asymmetric return-type precedence and incompatible helper synthesis. Updated `test_scope.py` test suite (`test_fixer_control_flow_and_side_effect_safety`) to pair equivalent subroutine units instead of a block with a whole function.
- **Fail-Closed Escaping Closure Protection**: Pre-unit closures, lambdas, and class methods that capture candidate output variables are treated as live downstream reads, regardless of whether the closure is called after the unit, escapes via callbacks, or is unreferenced downstream. This intentional fail-closed trade-off prevents unbound local errors at the expense of potential over-rejection.
- **Helper Return Type Inference**: Standardized `_infer_helper_return_type` signature to use explicit `unit_kind` and `is_subroutine` flags, ensuring accurate `-> Any` return type annotations for comprehension and complex expression units.
