# pyDoppelgangerHunt

AST structural code clone detector, redundancy gate, and refactoring patch synthesizer for Python.

[![CI](https://github.com/jonathanschaab/pyDoppelgangerHunt/actions/workflows/ci.yml/badge.svg)](https://github.com/jonathanschaab/pyDoppelgangerHunt/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-85.2%25-brightgreen.svg)](https://github.com/jonathanschaab/pyDoppelgangerHunt/pull/1)
[![Code Quality](https://img.shields.io/badge/pylint-10.00%2F10-brightgreen.svg)](https://github.com/jonathanschaab/pyDoppelgangerHunt/pull/1)
[![Clones](https://img.shields.io/badge/clones-0%20(dual--tier)-brightgreen.svg)](https://github.com/jonathanschaab/pyDoppelgangerHunt/pull/1)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![Dependencies: None](https://img.shields.io/badge/dependencies-0%20(stdlib%20only)-brightgreen.svg)](https://docs.python.org/3/)

`pyDoppelgangerHunt` analyzes Python Abstract Syntax Trees (AST) to detect Type-1 (exact), Type-2 (renamed identifiers), and Type-3 (gapped / modified statement) code duplicates. It operates with zero external runtime dependencies, relying exclusively on Python standard library modules.

---

## Technical Overview

### Detection Pipeline
1. **AST Extraction**: Parses functions, methods, compound blocks (`if`, `for`, `while`, `try`, `with`), and class bodies.
2. **Canonicalization & Normalization**:
   - Alpha-renaming of variables, attributes, and parameters (`VAR_0`, `VAR_1`) while preserving Python builtins.
   - Commutative operand sorting for symmetric operators (`a + b` $\to$ `b + a`, `x == y` $\to$ `y == x`).
   - Python idiom normalization (canonicalizing loops, comprehensions, and search idioms).
   - PEP 484/526 type annotation stripping.
3. **Similarity Metrics**:
   - $k$-shingle Jaccard similarity.
   - Multiset Jaccard (characteristic token vectors).
   - TF-IDF weighted token importance.
   - Longest Common Subsequence (LCS) alignment for gapped clone tolerance.
4. **Length & Prefix Bound Pruning**: Uses SourcererCC-style $O(1)$ upper-bound theorems to eliminate $>99\%$ of non-matching unit pairs prior to invoking dynamic programming.
5. **Clustering**: Groups transitive pairwise clone hits into connected component Clone Families using disjoint-set Union-Find with path compression.

---

## Installation

```bash
pip install pydoppelgangerhunt
```

Or clone directly as a git submodule:

```bash
git submodule add https://github.com/jonathanschaab/pyDoppelgangerHunt.git tools/pyDoppelgangerHunt
pip install -e ./tools/pyDoppelgangerHunt
```

---

## Usage Modes

### Command Line Interface

```bash
# Basic audit with 90% threshold and minimum 8 lines
pydoppelgangerhunt src/ --threshold 0.90 --min-lines 8

# Generate an interactive standalone HTML report
pydoppelgangerhunt src/ --html report.html --priority

# Sort clones by McCabe cyclomatic complexity priority score
pydoppelgangerhunt src/ --sort-by priority --top 10

# Audit git-modified lines only (PR diff gating)
pydoppelgangerhunt src/ --diff-only --since origin/main

# Include Jupyter Notebook (.ipynb) code cells
pydoppelgangerhunt notebooks/ --notebooks

# Detect divergent clones modified >90 days apart
pydoppelgangerhunt src/ --blame

# Cross-reference test coverage (SQLite .coverage or coverage.xml)
pydoppelgangerhunt src/ --coverage .coverage

# Generate git-apply compatible refactoring patch file
pydoppelgangerhunt src/ --patch refactor.patch

# Synthesize cross-file shared utility modules and replace duplicates
pydoppelgangerhunt src/ --patch refactor.patch --replace-clones --cross-file-strategy auto

# Emit OASIS SARIF 2.1.0 output for GitHub Code Scanning
pydoppelgangerhunt src/ --format sarif --output results.sarif
```

### GitHub Actions Integration

Add `pyDoppelgangerHunt` directly into your GitHub workflow:

```yaml
- name: Run pyDoppelgangerHunt Clone Gate
  uses: jonathanschaab/pyDoppelgangerHunt@v1
  with:
    target: "src/"
    threshold: "0.85"
    summary: true
    github_annotations: true
```

### Pre-Commit Hook

Add to your `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/jonathanschaab/pyDoppelgangerHunt
    rev: v1.0.0
    hooks:
      - id: pydoppelgangerhunt
        args: ["--diff-only"]
```

---

## Configuration

`pyDoppelgangerHunt` loads configuration from `pyproject.toml` or `.pydoppelgangerhunt.toml`:

```toml
[tool.pydoppelgangerhunt]
threshold = 0.85
min_lines = 8
min_tokens = 15
cross_file_strategy = "auto"
shared_module_name = "_common.py"
exclude = [
    "tests",
    "vendor",
    ".venv",
]
exemptions = [
    ["module_a.py:func_1", "module_b.py:func_2"],
]
```

### Cross-Module Deduplication Strategies

When refactoring clones across different files with `--patch` and `--replace-clones`, `pyDoppelgangerHunt` uses directed dependency graph analysis to prevent circular imports:

- **`auto`** *(default)*: Uses `shared_module` when clones share an enclosing Python package directory; falls back to `host_module` when clones only share the repository or `src/` root.
- **`shared_module`**: Synthesizes a shared helper into a common utility module (e.g. `_common.py`) and wires module imports for each caller.
- **`host_module`**: Extracts the helper into the primary clone file and wires callers to import from it.
- **`skip`**: Skips cross-module clone pairs and focuses exclusively on intra-file deduplication.

If any proposed cross-module extraction would introduce a circular import or unresolvable path, `pyDoppelgangerHunt` records a descriptive advisory comment and keeps the refactoring transactional without emitting broken imports.

### Programmatic Refactoring API

`pyDoppelgangerHunt` exposes a programmatic API for AST unit inspection, byte-to-character column translation, reverse-order buffer replacement, and multi-file patch synthesis:

```python
from pydoppelgangerhunt import (
    ReplacementItem,
    UnitCollisionError,
    UnitDict,
    UnitSpan,
    compute_unit_byte_offsets,
    compute_unit_char_offsets,
    compute_unit_replacement_span,
    compute_unit_spans,
    count_physical_newlines,
    detect_line_ending,
    generate_refactoring_patch,
    intervals_overlap,
    is_subroutine_unit,
    refactor_module_units,
    split_source_lines,
    validate_module_unit_replacements,
)
```

- **`UnitCollisionError`**: Subclasses `ValueError`. Raised when candidate replacement units collide or overlap within the same source buffer (either during Tier 1 semantic AST coordinate checks or Tier 2 physical byte sweeps). Because it inherits from `ValueError`, standard exception handlers catch it transparently, while specialized handlers can differentiate collision conditions.
- **`compute_unit_spans(source_text, unit)`**: Calculates exact 0-indexed character and UTF-8 byte spans for an AST unit, returning a structured `UnitSpan(start_char, end_char, start_byte, end_byte, is_column_bounded, start_line, end_line, start_col_char, end_col_char)`.
- **`compute_unit_byte_offsets(source_text, unit)`** & **`compute_unit_char_offsets(source_text, unit)`**: Fast convenience helpers returning `(start_byte, end_byte)` or `(start_char, end_char)` coordinate tuples.
- **`compute_unit_replacement_span(source_text, unit, replacement_text, preserve_boundary_pragmas=True)`**: Resolves a replacement into a `(start_char, end_char, final_replacement_text)` tuple against the unmodified source text while preserving attached `# type: ignore` or `# noqa` boundary pragmas.
- **`split_source_lines(source_text)`**: Splits source code into physical lines with line terminators preserved. Unlike `str.splitlines()`, it splits strictly on physical Python newline sequences (`\r\n`, `\r`, `\n`) and never on form feeds (`\f`) or vertical tabs (`\v`), matching Python grammar and AST coordinate semantics.
- **`count_physical_newlines(text)`**: Accurately counts physical line endings (`\r\n`, `\r`, `\n`) across mixed and legacy lone-CR formats.
- **`detect_line_ending(*sources)`**: Detects predominant line ending format across source strings or iterables via majority vote (`\n`, `\r\n`, or `\r`), breaking ties in priority order (`\n` → `\r\n` → `\r`).
- **`intervals_overlap(s1, e1, s2, e2)`**: Fast primitive returning `True` if two half-open intervals `[s1, e1)` and `[s2, e2)` intersect; `False` otherwise.
- **`is_subroutine_unit(unit)`**: Inspects unit metadata to determine if a clone unit represents
  an inner compound block, sliding window, or clause branch rather than an entire callable,
  applying heuristic fallbacks when `kind` is unspecified.
- **`refactor_module_units(source_text, replacements, tier1=True, tier2=True, dry_run=False)`**: Applies multiple non-overlapping unit replacements in strict **reverse source order** (descending byte offsets) using single-pass buffer slicing, guaranteeing that downstream text expansions or contractions never invalidate upstream coordinates.
- **`validate_module_unit_replacements(source_text, replacements, tier1=True, tier2=True)`**: Non-mutating validation helper that verifies candidate replacements for collisions across Tier 1 (AST coordinate overlap) and Tier 2 (physical byte interval sweep) without modifying or allocating new source string buffers.
- **`generate_refactoring_patch(candidate_pairs, repo_root=..., replace_clones=...)`**: Synthesizes a multi-file unified diff (`git apply` compatible) with dependency cycle detection and per-pair transactional snapshot rollback. For generator subroutines, downstream variable liveness and reaching definitions ensure output parameters are paired safely; pre-unit closures, lambdas, and class methods treat captured variables as live to protect escaping callbacks. When a unit is analyzed at top-level module scope, prior module functions contribute their global reads, causing un-definitely assigned candidates to fail closed safely (diagnostics emitted at DEBUG log level).

#### Dual-Tier Collision Validation & Pragma Preservation Example

```python
from pydoppelgangerhunt import (
    UnitCollisionError,
    check_units_overlap,
    refactor_module_units,
)

# 1. Tier 1 Semantic Overlap Rejection
# Two functions whose declared AST line spans overlap (lines 1-2 vs 2-3)
u1 = {"name": "fn1", "file": "mod.py", "start": 1, "end": 2}
u2 = {"name": "fn2", "file": "mod.py", "start": 2, "end": 3}

assert check_units_overlap(u1, u2) is True  # Fails fast before text resolution

# 2. Tier 2 Catch of Pragma-Expanded Collision
# In AST coordinates, u_stmt (col 0-5) and u_expr (col 7-21) do not overlap
src = "a = 1  # type: ignore\nb = 2\n"
u_stmt = {"name": "u_stmt", "file": "mod.py", "start": 1, "end": 1, "start_col": 0, "end_col": 5}
u_expr = {"name": "u_expr", "file": "mod.py", "start": 1, "end": 1, "start_col": 7, "end_col": 21}

assert check_units_overlap(u_stmt, u_expr) is False  # Tier 1 allows (disjoint columns)

# However, preserving the trailing '# type: ignore' pragma causes u_stmt's physical
# byte replacement span to expand across the full line, overlapping u_expr.
# Tier 2 catches the expanded byte interval collision:
try:
    refactor_module_units(src, [(u_stmt, "a = 10\n"), (u_expr, "val")])
except UnitCollisionError as exc:
    print(f"Tier 2 collision detected: {exc}")
```

#### Performance & Algorithmic Bounds

Refactoring executes in linear-logarithmic time with incremental validation:
- **Incremental Collision Checking**: Delegation checks candidate units incrementally ($O(K)$ per unit against existing replacements and physical items), eliminating expensive $O(K \log K)$ buffer dry-runs per clone pair.
- **Micro-Benchmark**: For modules with 100+ replacements and 10,000+ lines, dual-tier validation and single-pass descending buffer slicing executes in under 5 ms on modern hardware.

##### Performance Notes: Closure Scanning & Throughput

When extracting generator subroutine clones that assign variables, `pyDoppelgangerHunt` conducts
a pre-unit AST walk within the enclosing scope to detect closures, lambdas, or nested classes
that capture candidate outputs prior to unit execution. Because such closures may escape into
callback tables or event loops, strict mode treats these captured names as live, guaranteeing
fail-closed safety.

- **Memoization & Cache Invalidation**: Downstream liveness analysis caches AST traversal results
  in a thread-safe LRU cache keyed by SHA-256 source digest prefix, unit line/column coordinates,
  unit kind and name, candidate outputs, and unit modification timestamp (`mtime`, where available
  on unit dictionaries; source content digest provides definitive invalidation), eliminating
  duplicate traversals across identical clone boundaries.
- **Lenient Mode Bypass**: For large codebases or batch runs where callback-escaping closures are
  known not to occur, pre-unit closure scanning can be bypassed by specifying
  `--closure-strictness lenient` (or `--skip-pre-unit-closures`, or setting
  `closure_strictness = "lenient"` / `skip_pre_unit_closures = true` in `pyproject.toml`). When both
  CLI flags are supplied, `--closure-strictness` takes precedence over `--skip-pre-unit-closures`.

#### Safety Model & Fail-Closed Refactoring Guarantees

When synthesizing refactoring patches and shared helpers, `pyDoppelgangerHunt` enforces
strict fail-closed safety invariants:
- **Lexical Scope Containment & Pre-Unit Closure Isolation (Generator Subroutines)**: For generator
  subroutines, any closures, lambdas, generator expressions, or nested class definitions preceding a
  candidate unit within the enclosing lexical scope that capture potential output variables are
  conservatively treated as escaping reads. Even if a closure is not called directly within the
  unit's immediate block, it may have registered into callback tables or event loops. Candidate
  outputs captured by pre-unit closures are preserved or cause the pair to fail closed rather than
  risk silent state corruption. To bypass pre-unit closure scanning when closures are known not to
  escape, pass `--skip-pre-unit-closures` or `--closure-strictness lenient` (when both are supplied,
  `--closure-strictness` takes precedence).
- **Abnormal Exit Loss Prevention**: Generator subroutines lexically enclosed in `try` blocks whose
  `except` or `finally` handlers read needed outputs are rejected. In Python, abnormal generator
  termination (`gen.close()`, `throw()`, or an exception) causes `yield from` to exit abruptly
  without returning, bypassing assignment to output variables (e.g. `total = (yield from ...)`
  never assigns `total`), leaving exception/cleanup handlers with unassigned or stale values.
- **Dynamic Scope & Evaluation Caveats**: Static analysis tracks lexical scopes, closures, lambdas,
  and generator expressions. Dynamic constructs downstream or within the unit such as `eval()`,
  `exec()`, `locals()`, or `vars()` cannot be inspected statically and fall outside static liveness
  guarantees.
- **Coordinate Clamping**: Coordinates accept integer or colon format (`"start:end"` or
  `"line:col"`), with line numbers clamped to at least line 1 and end clamped to at least start.
- **Definite Assignment Verification**: Synthesized helper return values and tuple-unpacked
  subroutine outputs require definite assignments along all incoming and internal execution
  paths. If an output variable could remain unassigned before helper exit, refactoring is
  rejected to prevent `UnboundLocalError`.
- **Async Generator Value Prohibition**: In Python, asynchronous generators cannot combine
  `yield` with explicit `return <value>`. Any candidate clone pair where either unit is an async
  generator attempting to propagate return values or outputs is unconditionally skipped.
- **Dual-Tier Transactional Rollback**: If helper extraction or call replacement encounters
  semantic overlap, token collisions, or dependency graph cycles, the entire refactoring
  operation rolls back cleanly without leaving partial mutations or corrupting host source files.

### Troubleshooting: When the Patcher Refuses a Candidate Pair
If `pyDoppelgangerHunt` reports clones but does not propose extractions for a pair when running
with `--patch`, check verbose logs (`-v` or logging level `DEBUG`):
- **Pre-Unit Closure Escapes**: If candidate outputs are captured by closures or callbacks defined
  prior to the unit, run with `--skip-pre-unit-closures` or `--closure-strictness lenient`.
- **Abnormal Exit Handlers**: If a generator subroutine is enclosed in a `try` block whose `finally`
  or `except` handlers read needed outputs, extraction is rejected to protect cleanup integrity.
- **Indefinite Stores**: If an output variable lacks definite assignment along every internal
  path, the extraction is rejected to protect against runtime `UnboundLocalError`.
- **Unpaired Outputs**: When duplicate variable names or arity mismatches prevent 1-to-1 output
  pairing, the pair fails closed safely.

To generate a starter configuration file in your project root:

```bash
pydoppelgangerhunt --init
```

---

## CLI Reference

| Flag | Argument | Description |
| :--- | :--- | :--- |
| `target` | `[path]` | Directory or package to scan (default: `.`) |
| `--threshold` | `FLOAT` | Minimum similarity threshold between 0.0 and 1.0 (default: `0.90`) |
| `--min-lines` | `INT` | Minimum line count per code block (default: `8`) |
| `--min-tokens` | `INT` | Minimum normalized token count (default: `15`) |
| `--format` | `text\|json\|sarif` | Output report format (default: `text`) |
| `--output`, `-o` | `PATH` | Output file path to save report |
| `--html` | `PATH` | Write standalone interactive HTML dashboard report |
| `--patch` | `PATH` | Write git-apply compatible unified patch file |
| `--replace-clones` | Flag | Replace duplicate clone bodies with calls delegating to extracted helpers |
| `--type-merge-strategy` | `fallback_any\|union` | Parameter typing strategy for helper synthesis (`fallback_any` or `union`) |
| `--method-binding` | `auto\|method\|module` | Target helper binding strategy (`auto`, `method`, or `module`) |
| `--cross-file-strategy` | `auto\|shared_module\|host_module\|skip` | Cross-module deduplication strategy (`auto`, `shared_module`, `host_module`, or `skip`; default: `auto`) |
| `--shared-module-name` | `FILENAME` | Target filename for shared utility extractions (default: `_common.py`) |
| `--skip-pre-unit-closures` | Flag | Skip pre-unit closure scan during subroutine extraction |
| `--closure-strictness` | `strict\|lenient` | Closure strictness mode (precedes boolean flag) |
| `--sort-by` | `similarity\|priority\|sloc` | Sort clone hits (default: `similarity`) |
| `--priority` | Flag | Sort clones by Priority score: $\text{Sim} \times \text{SLOC} \times \text{Complexity}$ |
| `--top` | `INT` | Truncate report to top $N$ clone pairs |
| `--cluster` | Flag | Cluster pairwise hits into connected component Clone Families |
| `--diff` | Flag | Print unified line diffs between clone instances |
| `--suggest` | Flag | Synthesize shared helper function signatures |
| `--blame` | Flag | Audit git commit history and warn on divergent clones ($>90$ days) |
| `--coverage` | `PATH` | Path to `.coverage` (SQLite) or `coverage.xml` to detect asymmetric coverage |
| `--notebooks` | Flag | Parse and harvest code cells from Jupyter Notebooks (`.ipynb`) |
| `--diff-only` | Flag | Audit only lines modified in git (PR gating) |
| `--since` | `REF` | Git commit or branch reference to compare against for `--diff-only` |
| `--github-annotations` | Flag | Emit `::warning` workflow commands for GitHub PR diff tabs |
| `--stats` | Flag | Compute repository DRY score, SLOC, DLOC, and duplication metrics |
| `--summary` | `PATH` | Path to write GitHub Step Summary Markdown report |
| `--baseline` | `PATH` | Path to grandfathered baseline JSON file to suppress |
| `--record-baseline` | `PATH` | Path to record detected clones into baseline JSON file |
| `--prune-baseline` | Flag | Prune orphaned fingerprints from baseline JSON file |
| `--min-calibration-frequency` | `INT` | Document frequency threshold to retain shingle in calibration (default: `1`; recommend >= 2 for monorepos > 100k units) |
| `--novel-pair-budget` | `INT` | Candidate pair budget cap for novel shingles in differential scans (default: `10000`) |
| `--sliding-window` | Flag | Enable sliding statement window scanner |
| `--harvest-closures` | Flag | Harvest nested closures and inner functions |
| `--idioms` | Flag | Canonicalize Python idioms (loops, comprehensions, search loops) |
| `--abstract-expressions`| Flag | Abstract arithmetic expressions and condition tests (NiCad Type-3-2) |
| `--gapped-tolerance` | Flag | Compute LCS alignment with length-bound pruning |
| `--preserve-docstrings` | Flag | Preserve docstrings during AST token extraction (docstrings stripped by default) |
| `--preserve-annotations`| Flag | Preserve PEP 484/526 type annotations (annotations stripped by default) |
| `--max-index-frequency` | `FLOAT` | Inverted index frequency threshold to prune ubiquitous shingles (default: `0.25`) |
| `--workers` | `INT` | Number of worker processes for parallel AST harvesting |

### Baseline Calibration & Massive Monorepo Deployments

When running differential scans in CI pipelines (`--diff-only --baseline baseline.json`), `pyDoppelgangerHunt` reuses pre-computed corpus calibration metadata (shingle frequencies and corpus size) from the recorded baseline to provide fast, approximate IDF weighting and dynamic stop-shingle pruning bounds without having to recompute global repository frequency statistics from scratch. To guarantee that calibration never suppresses potentially valid clone candidates (preventing false negatives), candidate pruning is conservative when calibrated document frequencies are near the pruning cutoff.

For massive monorepos (> 100k AST units), `compute_corpus_calibration` retains shingles across the entire codebase. By default, singleton shingles (appearing in only 1 unit) are indexed. Setting `--min-calibration-frequency 2` (or configuring `min_calibration_frequency = 2` in `pyproject.toml`) discards singleton shingles during `--record-baseline`, dramatically compressing baseline JSON file size and in-memory footprint while providing high stop-shingle pruning efficiency with approximate IDF weighting. When `--min-calibration-frequency` is specified during `--record-baseline`, `pyDoppelgangerHunt` records detailed shingle pruning counts, compression ratios, and estimated payload savings directly inside the `corpus_calibration` metadata and prints summary compression metrics upon baseline generation to facilitate CI compression benchmarking across diverse codebase topologies. In addition, `--novel-pair-budget 10000` (configurable via `novel_pair_budget` in `pyproject.toml`) bounds the pairwise candidate comparison budget for novel shingles introduced during differential scans.


---

## Quality Gates & Invariant Standards

All pull requests and releases are validated against 7 automated quality gates defined centrally in `gates.toml`:

| Gate | Target | Tool / Command | Invariant Threshold |
| :--- | :--- | :--- | :--- |
| **Gate 1** | Type Safety | `mypy` | 0 errors across 18 source files |
| **Gate 2** | Code Quality | `pylint` | Strict **10.00 / 10.00** rating |
| **Gate 3** | Packaging Hygiene | `deptry` | 0 unused, missing, or transitive dependencies |
| **Gate 4** | AST Security | `bandit` | 0 security issues (`-ll`) |
| **Gate 5** | Supply Chain | `pip-audit` | 0 known CVEs in virtual environment |
| **Gate 6** | Unit Tests & Coverage | `pytest` + `pytest-cov` | 80 passed, **$\ge 85.0\%$ branch coverage** (85.92%) |
| **Gate 7** | Clone Barrier | `pydoppelgangerhunt` | 0 clones across Tier 1 ($\ge 70\%$, 8 lines) and Tier 2 ($\ge 80\%$, 6 lines) |

### Executing Quality Gates Locally

All gates can be executed locally in under 25 seconds via the shared runner or cross-platform wrappers:

```bash
# Python runner (driven by gates.toml)
python scripts/run_quality_gates.py

# Run a specific gate by ID or number
python scripts/run_quality_gates.py --gate type-safety
python scripts/run_quality_gates.py --gate tests-coverage

# POSIX shell wrapper (Linux / macOS)
./scripts/check_quality_gates.sh

# PowerShell wrapper (Windows)
pwsh -File .\scripts\check_quality_gates.ps1
```

For full quality gate metrics and verification logs, see [Pull Request #1](https://github.com/jonathanschaab/pyDoppelgangerHunt/pull/1).

---

## References & Acknowledgements

The design and detection pipeline of `pyDoppelgangerHunt` draw inspiration from foundational research in software clone detection and static code analysis:

- **SourcererCC** (Sajnani, H., Saini, V., Ossher, J., & Lopes, C. V., *SourcererCC: Scaling Code Clone Detection to Big-Code*, ICSE 2016): Inverted index candidate filtering, token frequency weighting, and length/prefix-bound pruning theorems that enable near-linear clone detection across large codebases.
- **NiCad** (Roy, C. K., & Cordy, J. R., *NICAD: Accurate Detection of Near-Miss Intentional Clones Using Flexible Pretty-Printing and Code Normalization*, ICPC 2008): AST structural normalization, blind identifier renaming, commutative operator canonicalization, and boilerplate statement filtering for Type-2 and Type-3 near-miss clones.
- **CCAligner** (Wang, P., Svajlenko, J., Wu, Y., Xu, Y., & Roy, C. K., *CCAligner: A Code Clone Detector for Finding Clones with Large Language Gaps*, ICSE 2018): Token-based sequence alignment algorithms using Longest Common Subsequence (LCS) with dynamic bound pruning for gapped clone tolerance.
- **McCabe Cyclomatic Complexity** (McCabe, T. J., *A Complexity Measure*, IEEE Transactions on Software Engineering, 1976): Complexity-weighted priority scoring ($\text{Similarity} \times \text{SLOC} \times \text{Cyclomatic Complexity}$) to elevate high-risk, cognitively complex clone pairs in refactoring workflows.
- **Non-Maximum Suppression (NMS)**: Geometric containment suppression of redundant sub-clones within larger compound clone blocks, adapted from spatial object detection.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
