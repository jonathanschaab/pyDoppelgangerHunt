# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **Subroutine Kind Parity Enforcement**: Unconditionally reject clone pairs between subroutine units (blocks, loops, comprehensions, complex expressions) and whole functions/methods across all scan modes (including report-only mode). This prevents asymmetric return-type precedence and incompatible helper synthesis.
- **Fail-Closed Escaping Closure Protection**: Pre-unit closures, lambdas, and class methods that capture candidate output variables are treated as live downstream reads, guaranteeing fail-closed safety even if the closure escapes or is executed via indirect callbacks.
- **Helper Return Type Inference**: Standardized `_infer_helper_return_type` signature to use explicit `unit_kind` and `is_subroutine` flags, ensuring accurate `-> Any` return type annotations for comprehension and complex expression units.
