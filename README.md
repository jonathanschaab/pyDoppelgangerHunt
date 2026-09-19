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
- **`shared_module`**: Synthesizes a shared helper into a common utility module (e.g. `_common.py`) and wires relative or absolute imports for each caller.
- **`host_module`**: Extracts the helper into the primary clone file and wires callers to import from it.
- **`skip`**: Skips cross-module clone pairs and focuses exclusively on intra-file deduplication.

If any proposed cross-module extraction would introduce a circular import or unresolvable path, `pyDoppelgangerHunt` records a descriptive advisory comment and keeps the refactoring transactional without emitting broken imports.

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
| `--sliding-window` | Flag | Enable sliding statement window scanner |
| `--harvest-closures` | Flag | Harvest nested closures and inner functions |
| `--idioms` | Flag | Canonicalize Python idioms (loops, comprehensions, search loops) |
| `--abstract-expressions`| Flag | Abstract arithmetic expressions and condition tests (NiCad Type-3-2) |
| `--gapped-tolerance` | Flag | Compute LCS alignment with length-bound pruning |
| `--preserve-docstrings` | Flag | Preserve docstrings during AST token extraction (docstrings stripped by default) |
| `--preserve-annotations`| Flag | Preserve PEP 484/526 type annotations (annotations stripped by default) |
| `--max-index-frequency` | `FLOAT` | Inverted index frequency threshold to prune ubiquitous shingles (default: `0.25`) |
| `--workers` | `INT` | Number of worker processes for parallel AST harvesting |

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
