"""Command Line Interface (CLI) entrypoint for pyDoppelgangerHunt."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from pydoppelgangerhunt.baseline import (
    BaselineFingerprints,
    compute_calibration_config_hash,
    filter_clones_by_baseline,
    load_baseline,
    prune_baseline,
    record_baseline,
    _derive_target_offsets,
    _safe_bool,
    _safe_index_frequency,
    _safe_int,
)
from pydoppelgangerhunt.clustering import cluster_clone_families
from pydoppelgangerhunt.config import (
    DEFAULT_EXCLUDES,
    init_tool_configuration,
    load_tool_config,
    normalize_path_string,
)
from pydoppelgangerhunt.canonical_path import CanonicalPathResolver
from pydoppelgangerhunt.coverage import check_asymmetric_coverage, read_coverage_data
from pydoppelgangerhunt.fixer import generate_refactoring_patch
from pydoppelgangerhunt.git_diff import (
    check_temporal_divergence,
    filter_clones_by_git_diff,
    get_git_head_commit,
    get_git_modified_files,
    get_git_modified_line_ranges,
    get_git_repo_root,
)
from pydoppelgangerhunt.matcher import compute_priority_score, scan_target
from pydoppelgangerhunt.metrics import compute_repository_dry_stats
from pydoppelgangerhunt.reporters import (
    COLOR_BOLD,
    COLOR_CYAN,
    COLOR_GREEN,
    COLOR_MAGENTA,
    COLOR_RED,
    COLOR_YELLOW,
    colorize,
    emit_structured_report,
    format_github_annotations,
    format_json_report,
    format_markdown_summary,
    format_sarif_report,
    generate_clone_diff,
    generate_html_report,
    supports_color,
    synthesize_refactoring_suggestion,
)


def build_arg_parser() -> argparse.ArgumentParser:  # pydoppelgangerhunt: ignore
    """Builds the comprehensive CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="pydoppelgangerhunt",
        description="pyDoppelgangerHunt: AST Structural Code Clone & Redundancy Gate",
    )
    parser.add_argument(
        "target",
        nargs="?",
        default=None,
        help="Directory or package to scan (default: configured target or current directory)",
    )
    parser.add_argument("--threshold", type=float, default=None, help="Minimum similarity threshold (0.0 - 1.0)")
    parser.add_argument("--min-lines", type=int, default=None, help="Minimum lines of code per function/block")
    parser.add_argument("--min-tokens", type=int, default=None, help="Minimum normalized AST tokens")
    parser.add_argument("--exclude", action="append", default=[], help="Patterns to exclude")
    parser.add_argument("--window-size", type=int, default=None, help="Statement window size for sliding window scanner (default: 5)")
    parser.add_argument("--min-expr-complexity", type=int, default=None, help="Minimum operator complexity for complex expression detection (default: 4)")
    parser.add_argument("--config", type=str, default=None, help="Path to TOML configuration file (defaults to pyproject.toml)")
    parser.add_argument("--format", choices=["text", "json", "sarif"], default="text", help="Output report format (default: text)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose diagnostic output")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output file path to save report")
    parser.add_argument("--html", type=str, default=None, help="Path to write standalone interactive HTML report")
    parser.add_argument("--patch", type=str, default=None, help="Path to write git-apply compatible refactoring patch file")
    parser.add_argument("--replace-clones", action="store_true", help="Replace clone bodies with calls to the synthesized helper in generated patches")
    parser.add_argument(
        "--type-merge-strategy",
        type=str,
        choices=["fallback_any", "union"],
        default=None,
        help="Type annotation merging strategy for shared helper parameter synthesis ('fallback_any' or 'union'; default: 'fallback_any')",
    )
    parser.add_argument("--sort-by", choices=["similarity", "priority", "sloc"], default="similarity", help="Sort clones by similarity, priority, or sloc (default: similarity)")
    parser.add_argument("--priority", action="store_true", help="Shortcut to sort clones by Priority score (Similarity * SLOC * Complexity)")
    parser.add_argument("--top", type=int, default=None, help="Truncate report to top N clone pairs")
    parser.add_argument("--blame", action="store_true", help="Audit git commit history and warn on temporally divergent clones (>90 days)")
    parser.add_argument("--coverage", type=str, default=None, help="Path to .coverage (SQLite) or coverage.xml to detect asymmetric test coverage")
    parser.add_argument(
        "--notebooks",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Include Jupyter Notebook (.ipynb) code cells in clone scan",
    )
    parser.add_argument("--github-annotations", action="store_true", help="Emit GitHub Actions workflow commands (::warning) for PR annotations")

    parser.add_argument("--suggest", action="store_true", help="Synthesize refactoring recommendations and helper function signatures")
    parser.add_argument("--diff", action="store_true", help="Display unified diff between cloned code blocks")
    parser.add_argument("--cluster", action="store_true", help="Group pairwise clones into connected component Clone Families using Union-Find")
    parser.add_argument(
        "--linkage",
        type=str,
        choices=["single", "complete", "quasi_complete", "average", "medoid"],
        default=None,
        help="Clustering linkage strategy for clone families ('single', 'complete', 'quasi_complete', 'average', 'medoid'; default: 'single')",
    )
    parser.add_argument(
        "--min-cluster-similarity",
        type=float,
        default=None,
        help="Minimum intra-cluster similarity floor for cluster admission/merging (default: match threshold)",
    )
    parser.add_argument(
        "--linkage-tolerance",
        type=float,
        default=None,
        help="Permissible similarity fluctuation margin below floor for complete/quasi-complete linkage (default: 0.0, or 0.05 for quasi_complete)",
    )
    parser.add_argument("--diff-only", action="store_true", help="Only audit lines modified in git (PR diff gating)")
    parser.add_argument("--since", type=str, default=None, help="Git reference / commit / branch to compare against for --diff-only (default: HEAD)")
    parser.add_argument(
        "--partial-hunk-policy",
        type=str,
        choices=["any", "major", "new"],
        default=None,
        help="Diff overlap policy for PR gating ('any': >=1 line, 'major': >=50%% lines, 'new': >=80%% lines; default: 'any')",
    )
    parser.add_argument(
        "--min-diff-overlap",
        type=float,
        default=None,
        help="Minimum fractional line overlap [0.0 - 1.0] for a clone block to be considered modified in git diff",
    )
    parser.add_argument("--stats", action="store_true", help="Compute repository DRY score, DLOC, and duplication metrics")
    parser.add_argument("--summary", type=str, default=None, help="Path to write GitHub Step Summary Markdown report")
    parser.add_argument("--init", action="store_true", help="Initialize pyDoppelgangerHunt configuration file")
    parser.add_argument("--baseline", type=str, default=None, help="Path to grandfathered clone baseline JSON file")
    parser.add_argument("--record-baseline", type=str, default=None, help="Path to record detected clones into baseline JSON file")
    parser.add_argument("--prune-baseline", action="store_true", help="Prune orphaned fingerprints from baseline JSON file")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes for parallel AST harvesting (default: 1)")
    parser.add_argument("--max-index-frequency", type=float, default=None, help="Inverted index frequency threshold to prune ubiquitous shingles (default: 0.25)")
    parser.add_argument("--min-corpus-units", type=int, default=None, help="Minimum corpus unit count before activating dynamic frequency stop-shingle pruning (default: 4)")
    parser.add_argument(
        "--method-binding",
        type=str,
        choices=["auto", "method", "module"],
        default=None,
        help="Target method binding strategy for patch refactoring ('auto', 'method', or 'module'; default: 'auto')",
    )
    parser.add_argument(
        "--cross-file-strategy",
        type=str,
        choices=["auto", "shared_module", "host_module", "skip"],
        default=None,
        help="Strategy for cross-module clone refactoring ('auto', 'shared_module', 'host_module', or 'skip'; default: 'auto')",
    )
    parser.add_argument(
        "--shared-module-name",
        type=str,
        default=None,
        help="Module filename for shared utility extractions (default: '_common.py')",
    )

    color_group = parser.add_mutually_exclusive_group()
    color_group.add_argument("--color", dest="color", action="store_true", default=None, help="Force colorized terminal output")
    color_group.add_argument("--no-color", dest="color", action="store_false", help="Disable colorized terminal output")

    bool_flags = [
        ("--functions-only", "Scan only whole functions (disables compound block detection)"),
        ("--sliding-window", "Enable sliding statement window scanner (Architecture B)"),
        ("--blind-indexing", "Enable sequential parameter/attribute indexing (NiCad/pyChase-style)"),
        ("--merge-subtrees", "Merge adjacent and overlapping window/subtree clone hits (Deckard-style)"),
        ("--complex-expressions", "Audit inline complex expressions with high operator density"),
        ("--clause-level", "Audit branch- and handler-level logic (if/else bodies and try/except handlers)"),
        ("--data-tables", "Audit module-level dictionary, list, set, and tuple configuration tables"),
        ("--strip-annotations", "Strip PEP 484/526 type annotations during AST shingling (deprecated: enabled by default)"),
        ("--strip-docstrings", "Strip docstrings during AST token extraction (default: True)"),
        ("--nms", "Apply Non-Maximum Suppression to eliminate redundant sub-clones"),
        ("--class-level", "Audit classes (ast.ClassDef) for structural duplicates (PyChase-style)"),
        ("--blind-literals", "Normalize constants to LITERAL and strip docstrings (PyChase/NiCad-style)"),
        ("--bag-of-tokens", "Use multiset characteristic vectors for permutation invariance (Deckard-style)"),
        ("--filter-boilerplate", "Filter out logging, print, and assert telemetry before comparison (NiCad-style)"),
        ("--consistent-renaming", "Enable scoped def-use consistent alpha-renaming preserving builtins (NiCad-style)"),
        ("--tfidf", "Enable TF-IDF token/shingle weighting across corpus (SourcererCC/Alain-style)"),
        ("--harvest-closures", "Harvest nested closures and inner functions with parent qualnames (PyChase-style)"),
        ("--commutative", "Canonicalize commutative operands in boolean, arithmetic, and equality expressions (NiCad-style)"),
        ("--comprehensions", "Audit list, dict, set comprehensions and generator pipelines (PyChase-style)"),
        ("--idioms", "Canonicalize Python idioms (loops to comprehensions, search loops to any/all) (PyChase-style)"),
        ("--abstract-expressions", "Abstract arithmetic expressions and condition tests to generic tokens (NiCad Type-3-2)"),
        ("--gapped-tolerance", "Compute CCAligner-style Longest Common Subsequence alignment for gapped clone tolerance"),
        ("--call-sequences", "Audit function and method call traces for procedural pipeline duplicates"),
        ("--audit-tests", "Audit test suites for clone patterns and @pytest.mark.parametrize opportunities"),
        ("--stop-shingles", "Filter canonical boilerplate shingles (logging, main guards) to reduce spurious candidate pairs"),
    ]
    for flag_name, help_text in bool_flags:
        parser.add_argument(flag_name, action=argparse.BooleanOptionalAction, default=None, help=help_text)

    parser.add_argument("--preserve-annotations", action="store_true", default=None, help="Preserve PEP 484/526 type annotations during AST shingling")
    parser.add_argument("--preserve-docstrings", action="store_true", default=None, help="Preserve docstrings during AST token extraction (docstrings stripped by default)")
    parser.add_argument("--strict-type4", action="store_true", default=False, help="Fail with non-zero exit code if Type-4 semantic issues are found")
    parser.add_argument("--type4", "--semantic", action="store_true", help="Also run Type-4 semantic clone & consistency audit")

    return parser


def _write_artifact_file(dest_path: str, content: str, label: str, verbose: bool = True) -> None:
    """Writes report content to disk and emits console confirmation."""
    out_p = Path(dest_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as fh:
        fh.write(content)
    if verbose:
        print(f"[{label}] Saved to {dest_path}")


def _audit_clone_risk_warnings(
    u1: Dict[str, Any],
    u2: Dict[str, Any],
    *,
    audit_blame: bool = False,
    cov_data: Optional[Dict[str, Set[int]]] = None,
    use_color: bool = False,
    indent: str = "    ",
    repo_root: Optional[str] = None,
) -> List[str]:
    """Audits temporal divergence and asymmetric test coverage risks for a clone pair."""
    lines: List[str] = []
    if audit_blame:
        div = check_temporal_divergence(u1, u2, repo_root=repo_root)
        if div:
            msg = f"{indent}[WARN] Divergent clone risk: {div['divergence_days']} days difference between edits!"
            print(colorize(msg, COLOR_YELLOW, use_color))
            lines.append(msg)
    if cov_data:
        asym = check_asymmetric_coverage(u1, u2, cov_data)
        if asym:
            c1, c2 = asym
            f1 = normalize_path_string(str(u1.get("file") or ""), strip_anchor=False)
            f2 = normalize_path_string(str(u2.get("file") or ""), strip_anchor=False)
            msg = f"{indent}[WARN] Asymmetric test coverage: {f1} ({c1:.0%}) vs {f2} ({c2:.0%})"
            print(colorize(msg, COLOR_YELLOW, use_color))
            lines.append(msg)
    return lines


def _run_type4_semantic_audit(
    target: str,
    excludes: List[str],
    strict_type4: bool,
    use_color: bool,
) -> bool:
    """Runs optional Type-4 semantic clone checks, returning True if strict check failed."""
    try:
        import importlib
        mod = importlib.import_module("check_semantic_clones")
        find_semantic_clones = getattr(mod, "find_semantic_clones")
        report_semantic_results = getattr(mod, "report_semantic_results")
        print(f"Scanning '{target}' for Type-4 Semantic Clones & Consistency Violations...")
        results = find_semantic_clones(target, excludes=excludes)
        has_violations = report_semantic_results(results)
        if has_violations and strict_type4:
            print(colorize("[FAIL] Type-4 semantic clone violations detected!", COLOR_BOLD + COLOR_RED, use_color))
            return True
    except (ImportError, AttributeError):
        print("[INFO] check_semantic_clones not found; skipping Type-4 scan.")
    return False


def _format_scan_mode_description(
    args: argparse.Namespace,
    *,
    strip_annotations: bool,
    strip_docstrings: bool,
    idioms_enabled: bool,
    call_seq_enabled: bool,
    audit_tests_enabled: bool,
    stop_shingles_enabled: bool,
) -> str:
    """Builds human-readable description string of active AST scanning modes."""
    modes: List[str] = ["functions"]
    if not args.functions_only:
        modes.append("compound blocks")
    flag_modes = [
        (args.sliding_window, "sliding windows"),
        (args.complex_expressions, "complex expressions"),
        (args.clause_level, "clause branches"),
        (args.data_tables, "data tables"),
        (args.class_level, "classes"),
        (args.merge_subtrees, "merged subtrees"),
        (args.blind_indexing, "blind indexed"),
        (strip_annotations, "untyped"),
        (strip_docstrings, "docstrings stripped"),
        (args.blind_literals, "blind literals"),
        (args.bag_of_tokens, "bag of tokens"),
        (args.filter_boilerplate, "filtered boilerplate"),
        (args.consistent_renaming, "consistent renaming"),
        (args.tfidf, "tfidf weighted"),
        (args.harvest_closures, "closure harvesting"),
        (args.commutative, "commutative"),
        (args.comprehensions, "comprehensions"),
        (idioms_enabled, "idioms canonicalized"),
        (args.abstract_expressions, "abstract expressions"),
        (args.gapped_tolerance, "gapped tolerance"),
        (call_seq_enabled, "call sequences"),
        (audit_tests_enabled, "audit tests"),
        (stop_shingles_enabled, "stop-shingles filtered"),
        (args.nms, "nms suppressed"),
    ]
    for is_enabled, label in flag_modes:
        if is_enabled:
            modes.append(label)
    return " + ".join(modes)


def _render_dry_scorecard(stats: Dict[str, Any], use_color: bool) -> None:
    """Emits ASCII scorecard and duplication summary to console."""
    print("\n" + "=" * 55)
    grade = str(stats.get("grade", "A+"))
    grade_color = COLOR_GREEN if grade in ("A+", "A") else (COLOR_YELLOW if grade in ("B", "C") else COLOR_RED)
    score_str = colorize(f"{float(stats.get('dry_score', 100.0)):.1f}% (Grade: {grade})", COLOR_BOLD + grade_color, use_color)
    print(f"pyDoppelgangerHunt DRY Scorecard: {score_str}")
    print(f"SLOC: {int(stats.get('sloc', 0)):,} | DLOC: {int(stats.get('dloc', 0)):,} | Duplication: {float(stats.get('duplication_pct', 0.0)):.2f}%")
    print(f"Clone Pairs: {int(stats.get('clone_pairs', 0))} | Clone Families: {int(stats.get('clone_families', 0))}")
    print("=" * 55)


def _render_pair_diff_and_suggestions(
    u1: Dict[str, Any],
    u2: Dict[str, Any],
    *,
    args: argparse.Namespace,
    target_repo_root: str,
    use_color: bool,
    indent: str = "    ",
) -> List[str]:
    """Renders optional refactoring suggestion and unified diff for a pair of clone units."""
    lines: List[str] = []
    if args.suggest:
        sug = synthesize_refactoring_suggestion(u1, u2, repo_root=target_repo_root)
        sug_colored = colorize(sug, COLOR_YELLOW, use_color)
        print(indent + sug_colored.replace("\n", "\n" + indent))
        lines.append(indent + sug.replace("\n", "\n" + indent))
    if args.diff:
        diff_out = generate_clone_diff(
            u1, u2, repo_root=target_repo_root, color=use_color
        )
        if diff_out:
            print(f"{indent}--- Diff ---")
            print(indent + diff_out.replace("\n", "\n" + indent))
            lines.append(f"{indent}--- Diff ---\n{indent}" + diff_out.replace("\n", "\n" + indent))
    return lines


def _safe_call_git_diff_helper(
    fn: Any,
    since_ref: Optional[str],
    repo_root: Optional[str],
) -> Any:
    """Invokes git diff helper function supporting optional repo_root parameter."""
    try:
        return fn(since_ref=since_ref, repo_root=repo_root)
    except TypeError:
        return fn(since_ref=since_ref)


def _resolve_target_relative_path(
    p_str: str,
    res_git: Path,
    res_target: Path,
) -> Optional[str]:
    """Attempts to resolve a git-worktree-relative path relative to a target directory root."""
    resolver = CanonicalPathResolver(target_root=res_target, repo_root=res_git)
    cp = resolver.resolve(p_str, basis="repo")
    return cp.target_relative


def _normalize_git_paths_for_target(
    raw_paths: Sequence[str],
    git_root: str,
    target_repo_root: str,
) -> List[str]:
    """Normalizes worktree-relative Git paths to target-relative paths when targeting a subdirectory."""
    try:
        res_git = Path(git_root).resolve()
        res_target = Path(target_repo_root).resolve()
    except (ValueError, OSError, RuntimeError):
        return list(raw_paths)
    if res_git == res_target:
        return [p for p in raw_paths if p]

    normalized: List[str] = []
    for p_str in raw_paths:
        if not p_str:
            continue
        rel = _resolve_target_relative_path(p_str, res_git, res_target)
        if rel and rel != ".":
            if p_str not in normalized:
                normalized.append(p_str)
    return normalized


def _normalize_modified_ranges_for_target(
    modified_ranges: Dict[str, List[Tuple[int, int]]],
    git_root: str,
    target_repo_root: str,
) -> Dict[str, List[Tuple[int, int]]]:
    """Expands modified ranges to include target-relative paths alongside worktree-relative paths."""
    try:
        res_git = Path(git_root).resolve()
        res_target = Path(target_repo_root).resolve()
    except (ValueError, OSError, RuntimeError):
        return modified_ranges
    if res_git == res_target:
        return modified_ranges

    target_ranges: Dict[str, List[Tuple[int, int]]] = {}
    for p_str, ranges in modified_ranges.items():
        if not p_str:
            continue
        rel = _resolve_target_relative_path(p_str, res_git, res_target)
        if rel and rel != ".":
            target_ranges[rel] = ranges
            target_ranges[p_str] = ranges
    return target_ranges


def _apply_baseline_and_diff_filters(
    clones: List[Tuple[float, Dict[str, Any], Dict[str, Any]]],
    args: argparse.Namespace,
    tool_cfg: Dict[str, Any],
    target_repo_root: str,
    preloaded_baseline: Optional[BaselineFingerprints] = None,
    target: Optional[str] = None,
) -> Tuple[List[Tuple[float, Dict[str, Any], Dict[str, Any]]], Optional[int]]:
    """Applies baseline pruning, baseline suppression, and git diff line filtering.

    Returns:
        Tuple of (filtered_clones, early_exit_code). If early_exit_code is not None, execution should terminate.
    """
    baseline_path = args.baseline or tool_cfg.get("baseline")

    target_val = target or getattr(args, "target", None) or target_repo_root
    git_root = _safe_call_git_diff_helper(get_git_repo_root, None, target_repo_root)
    git_worktree_root = git_root if git_root and os.path.exists(git_root) else target_repo_root

    if args.prune_baseline:
        if not baseline_path:
            print("[ERROR] --prune-baseline requires a baseline path (specify via --baseline or config)")
            return clones, 1
        if not os.path.exists(baseline_path):
            print(f"[ERROR] Baseline file '{baseline_path}' not found")
            return clones, 1
        try:
            prune_res = prune_baseline(
                baseline_path,
                clones,
                repo_root=git_worktree_root,
                target=target_val,
                clone_basis="target_relative",
            )
        except TypeError:
            try:
                prune_res = prune_baseline(
                    baseline_path, clones, repo_root=git_worktree_root, target=target_val
                )
            except TypeError:
                prune_res = prune_baseline(baseline_path, clones)
        pruned_count = prune_res[0]
        retained_count = prune_res[1]
        skipped_dirty = getattr(prune_res, "skipped_dirty_count", 0)
        if args.format == "text":
            if skipped_dirty > 0:
                print(
                    f"[BASELINE] Pruned {pruned_count} orphaned fingerprint(s) from {baseline_path} "
                    f"({skipped_dirty} skipped due to unstaged git changes, {retained_count} retained)."
                )
            else:
                print(
                    f"[BASELINE] Pruned {pruned_count} orphaned fingerprint(s) from {baseline_path} "
                    f"({retained_count} retained)."
                )

    if baseline_path:
        if preloaded_baseline is not None and not args.prune_baseline:
            base_fps = preloaded_baseline
        else:
            base_fps = load_baseline(baseline_path)
        try:
            clones, suppressed_count = filter_clones_by_baseline(
                clones,
                base_fps,
                repo_root=git_worktree_root,
                target=target_val,
                clone_basis="target_relative",
            )
        except TypeError:
            clones, suppressed_count = filter_clones_by_baseline(
                clones, base_fps, repo_root=git_worktree_root, target=target_val
            )
        if args.format == "text":
            print(f"[BASELINE] Suppressed {suppressed_count} grandfathered clone(s). {len(clones)} un-grandfathered clone(s) remaining.")

    if args.diff_only:
        diff_policy = args.partial_hunk_policy or str(tool_cfg.get("partial_hunk_policy", "any"))
        min_overlap = (
            args.min_diff_overlap
            if args.min_diff_overlap is not None
            else float(tool_cfg.get("min_diff_overlap", 0.0))
        )
        git_diff_root = git_worktree_root
        modified_ranges = _safe_call_git_diff_helper(
            get_git_modified_line_ranges, args.since, git_diff_root
        )
        if modified_ranges and git_diff_root != target_repo_root:
            modified_ranges = _normalize_modified_ranges_for_target(
                modified_ranges, git_diff_root, target_repo_root
            )
        clones = filter_clones_by_git_diff(
            clones,
            modified_ranges,
            policy=diff_policy,
            min_overlap_ratio=min_overlap,
        )
        if args.format == "text":
            print(f"[INFO] Filtered by git diff (policy={diff_policy}): {len(clones)} clone pair(s) touch modified lines.")

    return clones, None


def _render_text_violations(
    clones: List[Tuple[float, Dict[str, Any], Dict[str, Any]]],
    threshold: float,
    sort_by: str,
    *,
    args: argparse.Namespace,
    families: Optional[List[Dict[str, Any]]] = None,
    cov_data: Optional[Dict[str, Set[int]]] = None,
    use_color: bool = False,
    target_repo_root: str = ".",
    audit_tests_enabled: bool = False,
) -> List[str]:
    """Renders human-readable text output for clone violations (either clustered families or pairwise)."""
    report_lines: List[str] = []
    active_cov = cov_data or {}
    if args.cluster and families is not None:
        title = f"\n[VIOLATION] Clustered into {len(families)} Clone Family/Families (from {len(clones)} pairwise hits) >= {threshold:.0%}:\n"
        print(colorize(title, COLOR_BOLD + COLOR_RED, use_color))
        report_lines.append(title)
        for fam in families:
            fam_badge = colorize(f"[{fam['family_id']}]", COLOR_BOLD + COLOR_MAGENTA, use_color)
            coherence_str = f", {fam['coherence']:.1%} coherence" if "coherence" in fam else ""
            f_head = (
                f"  * {fam_badge} {fam['member_count']} members "
                f"(avg sim {fam['avg_similarity']:.1%}{coherence_str}, max {fam['max_similarity']:.1%}, "
                f"{fam['total_lines']} lines across {len(fam['unique_files'])} file(s)):"
            )
            print(f_head)
            report_lines.append(f_head)
            medoid_name = (
                str(fam["medoid"].get("name") or "")
                if isinstance(fam.get("medoid"), dict)
                else None
            )
            for m in fam.get("members", []):
                m_file = normalize_path_string(str(m.get("file") or ""), strip_anchor=False)
                m_start = int(m.get("start") or 1)
                m_end = int(m.get("end") or m_start)
                m_name = str(m.get("name") or "member")
                m_tag = " [medoid]" if medoid_name and m_name == medoid_name else ""
                m_line = f"      - {m_file}:{m_start}-{m_end} ({m_name}){m_tag}"
                print(m_line)
                report_lines.append(m_line)
            if len(fam["members"]) >= 2:
                report_lines.extend(
                    _audit_clone_risk_warnings(
                        fam["members"][0],
                        fam["members"][1],
                        audit_blame=args.blame,
                        cov_data=active_cov,
                        use_color=use_color,
                        indent="      ",
                        repo_root=target_repo_root,
                    )
                )
                report_lines.extend(
                    _render_pair_diff_and_suggestions(
                        fam["members"][0],
                        fam["members"][1],
                        args=args,
                        target_repo_root=target_repo_root,
                        use_color=use_color,
                        indent="      ",
                    )
                )
    else:
        title = f"\n[VIOLATION] Found {len(clones)} AST structural clone pair(s) >= {threshold:.0%}:\n"
        print(colorize(title, COLOR_BOLD + COLOR_RED, use_color))
        report_lines.append(title)
        for sim, u1, u2 in clones:
            sim_badge = colorize(f"[{sim:.1%}]", COLOR_BOLD + COLOR_CYAN, use_color)
            prefix = f"  * {sim_badge}"
            if sort_by == "priority":
                p_val = compute_priority_score(sim, u1, u2)
                p_badge = colorize(f"[Priority {p_val:.1f}]", COLOR_BOLD + COLOR_MAGENTA, use_color)
                prefix = f"  * {sim_badge} {p_badge}"
            f1 = normalize_path_string(str(u1.get("file") or ""), strip_anchor=False)
            f2 = normalize_path_string(str(u2.get("file") or ""), strip_anchor=False)
            s1 = int(u1.get("start") or 1)
            e1 = int(u1.get("end") or s1)
            s2 = int(u2.get("start") or 1)
            e2 = int(u2.get("end") or s2)
            n1 = str(u1.get("name") or "unit1")
            n2 = str(u2.get("name") or "unit2")
            line = f"{prefix} {f1}:{s1}-{e1} ({n1}) <===> {f2}:{s2}-{e2} ({n2})"
            print(line)
            report_lines.append(line)
            report_lines.extend(
                _audit_clone_risk_warnings(
                    u1,
                    u2,
                    audit_blame=args.blame,
                    cov_data=active_cov,
                    use_color=use_color,
                    indent="    ",
                    repo_root=target_repo_root,
                )
            )
            if audit_tests_enabled and n1.startswith("test_") and n2.startswith("test_"):
                tip = colorize("    [TIP] Consider refactoring with @pytest.mark.parametrize", COLOR_YELLOW, use_color)
                print(tip)
                report_lines.append("    [TIP] Consider refactoring with @pytest.mark.parametrize")
            report_lines.extend(
                _render_pair_diff_and_suggestions(
                    u1,
                    u2,
                    args=args,
                    target_repo_root=target_repo_root,
                    use_color=use_color,
                    indent="    ",
                )
            )

    footer = "\nPlease refactor structural duplicates into shared helpers, base models, or declarative specifications."
    print(footer)
    report_lines.append(footer)
    return report_lines


def _resolve_strip_option(
    args: argparse.Namespace,
    tool_cfg: Dict[str, Any],
    active_calib: Dict[str, Any],
    has_explicit_cfg: bool,
    strip_name: str,
    preserve_name: str,
) -> bool:
    """Resolves strip/preserve boolean flags with precedence: CLI preserve -> CLI strip -> config -> calibration -> default."""
    if getattr(args, preserve_name, None):
        return False
    cli_strip = getattr(args, strip_name, None)
    if cli_strip is not None:
        return bool(cli_strip)
    if has_explicit_cfg and strip_name in tool_cfg:
        return bool(tool_cfg[strip_name])
    if has_explicit_cfg and preserve_name in tool_cfg:
        return not bool(tool_cfg[preserve_name])
    if strip_name in active_calib:
        return _safe_bool(active_calib[strip_name])
    return bool(tool_cfg.get(strip_name, not tool_cfg.get(preserve_name, False)))


def _resolve_frequency_option(
    cli_arg: Optional[float],
    tool_cfg: Dict[str, Any],
    active_calib: Dict[str, Any],
    has_explicit_cfg: bool,
) -> Optional[float]:
    """Resolves index frequency with precedence: CLI -> explicit config -> calibration -> implicit config -> default (0.25)."""
    if cli_arg is not None:
        return cli_arg
    if has_explicit_cfg and "max_index_frequency" in tool_cfg:
        cfg_val = tool_cfg["max_index_frequency"]
        return _safe_index_frequency(cfg_val) if cfg_val is not None else None
    if "max_index_frequency" in active_calib:
        calib_val = active_calib["max_index_frequency"]
        if calib_val is None:
            return None
        parsed_val = _safe_index_frequency(calib_val)
        return parsed_val if parsed_val is not None else 0.25
    if "max_index_frequency" in tool_cfg:
        implicit_cfg = tool_cfg["max_index_frequency"]
        return _safe_index_frequency(implicit_cfg) if implicit_cfg is not None else None
    return 0.25


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Main execution CLI entrypoint."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    target_arg = args.target
    if args.init:
        init_dir = target_arg if (target_arg and os.path.isdir(target_arg)) else "."
        cfg_target = init_tool_configuration(init_dir)
        print(f"[OK] Initialized pyDoppelgangerHunt configuration at {cfg_target}")
        return 0

    use_color = supports_color(args.color)

    # Load configuration from pyproject.toml / config file
    tool_cfg: Dict[str, Any] = {}
    if args.config:
        tool_cfg = load_tool_config(args.config)
    else:
        if target_arg:
            target_dir = (
                target_arg
                if os.path.isdir(target_arg)
                else (os.path.dirname(target_arg) or ".")
            )
            if target_dir != ".":
                try:
                    tool_cfg = load_tool_config(repo_root=target_dir)
                except TypeError:
                    pass
        if not tool_cfg:
            tool_cfg = load_tool_config()
    cfg_target = str(tool_cfg.get("target") or "")
    default_dir = (
        cfg_target
        if cfg_target and os.path.isdir(cfg_target)
        else (
            "pydoppelgangerhunt"
            if os.path.isdir("pydoppelgangerhunt")
            else ("src" if os.path.isdir("src") else ".")
        )
    )
    target = target_arg or default_dir
    target_dir = target if os.path.isdir(target) else (os.path.dirname(target) or ".")
    target_repo_root = target_dir
    git_root = _safe_call_git_diff_helper(get_git_repo_root, None, target_repo_root)
    git_worktree_root = git_root if git_root and os.path.exists(git_root) else target_repo_root
    threshold = args.threshold if args.threshold is not None else float(tool_cfg.get("threshold", 0.90))
    cfg_exemptions = (
        [tuple(p) for p in tool_cfg["exemptions"]] if "exemptions" in tool_cfg else None
    )

    raw_binding = args.method_binding or str(tool_cfg.get("method_binding", "auto"))
    method_binding = raw_binding if raw_binding in ("auto", "method", "module") else "auto"
    replace_clones = args.replace_clones or bool(tool_cfg.get("replace_clones", False))
    raw_strategy = args.type_merge_strategy or str(tool_cfg.get("type_merge_strategy", "fallback_any"))
    type_merge_strategy = raw_strategy if raw_strategy in ("fallback_any", "union") else "fallback_any"
    raw_cross_file = args.cross_file_strategy or str(tool_cfg.get("cross_file_strategy", "auto"))
    cross_file_strategy = (
        raw_cross_file
        if raw_cross_file in ("auto", "shared_module", "host_module", "host", "shared", "skip")
        else "auto"
    )
    shared_module_name = str(args.shared_module_name or tool_cfg.get("shared_module_name", "_common.py"))

    baseline_path = args.baseline or tool_cfg.get("baseline")
    preloaded_baseline: Optional[BaselineFingerprints] = None
    calib_dict: Optional[Dict[str, Any]] = None
    scan_offset: Optional[str] = None
    if baseline_path and not args.record_baseline and os.path.exists(baseline_path):
        preloaded_baseline = load_baseline(baseline_path)
        calib_dict = getattr(preloaded_baseline, "corpus_calibration", None)
        base_target = getattr(preloaded_baseline, "target", None)
        base_target_rel = getattr(preloaded_baseline, "target_repo_relative", None)
        target_val = target or getattr(args, "target", None) or target_repo_root
        base_offset, scan_offset = _derive_target_offsets(
            base_target, base_target_rel, git_worktree_root, target_val
        )
        if calib_dict and isinstance(calib_dict, dict):
            if (base_offset or "") != (scan_offset or ""):
                if args.format == "text":
                    b_desc = base_offset if base_offset else "repository root"
                    s_desc = scan_offset if scan_offset else "repository root"
                    print(
                        colorize(
                            f"Info: Active scan scope ({s_desc}) differs from baseline calibration scope ({b_desc}). "
                            f"Skipping baseline corpus calibration reuse to prevent shingle frequency skew.",
                            COLOR_YELLOW,
                            use_color,
                        )
                    )
                calib_dict = None

    active_calib: Dict[str, Any] = dict(calib_dict) if (calib_dict and isinstance(calib_dict, dict)) else {}
    has_explicit_cfg = bool(args.config)

    max_index_frequency = _resolve_frequency_option(
        args.max_index_frequency, tool_cfg, active_calib, has_explicit_cfg
    )

    int_specs: Sequence[Tuple[str, str, Optional[int], int, Optional[int]]] = (
        ("min_lines", "min_lines", args.min_lines, 1, 8),
        ("min_tokens", "min_tokens", args.min_tokens, 1, 15),
        ("min_corpus_size", "min_corpus_units", args.min_corpus_units, 0, None),
        ("window_size", "window_size", args.window_size, 1, 5),
        ("min_expr_complexity", "min_expr_complexity", args.min_expr_complexity, 1, 4),
    )
    resolved_ints: Dict[str, Optional[int]] = {}
    for calib_key, cfg_key, cli_arg, min_bound, default_val in int_specs:
        if cli_arg is not None:
            resolved_ints[calib_key] = cli_arg
        elif has_explicit_cfg and cfg_key in tool_cfg:
            resolved_ints[calib_key] = _safe_int(tool_cfg[cfg_key], min_val=min_bound)
        elif calib_key in active_calib:
            resolved_ints[calib_key] = _safe_int(active_calib[calib_key], min_val=min_bound)
        else:
            raw_cfg = tool_cfg.get(cfg_key)
            parsed_cfg = _safe_int(raw_cfg, min_val=min_bound) if raw_cfg is not None else None
            resolved_ints[calib_key] = parsed_cfg if parsed_cfg is not None else default_val

    min_lines = resolved_ints.get("min_lines", 8) or 8
    min_tokens = resolved_ints.get("min_tokens", 15) or 15
    min_corpus_size = resolved_ints.get("min_corpus_size")
    window_size = resolved_ints.get("window_size", 5) or 5
    min_expr_complexity = resolved_ints.get("min_expr_complexity", 4) or 4
    effective_window_size = window_size
    effective_min_expr_complexity = min_expr_complexity

    if not args.exclude and (not has_explicit_cfg or "exclude" not in tool_cfg) and "excludes" in active_calib:
        raw_ex = active_calib.get("excludes")
        if isinstance(raw_ex, (list, tuple, set)):
            excludes = [str(x) for x in raw_ex]
        else:
            excludes = list(tool_cfg.get("exclude", DEFAULT_EXCLUDES))
    else:
        excludes = list(tool_cfg.get("exclude", DEFAULT_EXCLUDES)) + args.exclude

    strip_specs = (
        ("strip_annotations", "preserve_annotations"),
        ("strip_docstrings", "preserve_docstrings"),
    )
    for s_name, p_name in strip_specs:
        s_val = _resolve_strip_option(args, tool_cfg, active_calib, has_explicit_cfg, s_name, p_name)
        setattr(args, s_name, s_val)
    strip_annotations = bool(args.strip_annotations)
    strip_docstrings = bool(args.strip_docstrings)

    bool_specs: Sequence[Tuple[str, str, str, bool]] = (
        ("bag_of_tokens", "bag_of_tokens", "bag_of_tokens", False),
        ("call_sequences", "call_sequences", "call_sequences", False),
        ("filter_stop_shingles", "stop_shingles", "stop_shingles", False),
        ("audit_tests", "audit_tests", "audit_tests", False),
        ("include_notebooks", "notebooks", "notebooks", False),
        ("blind_literals", "blind_literals", "blind_literals", False),
        ("idioms", "idioms", "idioms", False),
        ("commutative", "commutative", "commutative", False),
        ("filter_boilerplate", "filter_boilerplate", "filter_boilerplate", False),
        ("consistent_renaming", "consistent_renaming", "consistent_renaming", False),
        ("abstract_expressions", "abstract_expressions", "abstract_expressions", False),
        ("blind_indexing", "blind_indexing", "blind_indexing", False),
        ("functions_only", "functions_only", "functions_only", False),
        ("class_level", "class_level", "class_level", False),
        ("sliding_window", "sliding_window", "sliding_window", False),
        ("clause_level", "clause_level", "clause_level", False),
        ("data_tables", "data_tables", "data_tables", False),
        ("harvest_closures", "harvest_closures", "harvest_closures", False),
        ("comprehensions", "comprehensions", "comprehensions", False),
        ("complex_expressions", "complex_expressions", "complex_expressions", False),
    )
    for calib_key, cli_attr, cfg_key, default_val in bool_specs:
        cli_val = getattr(args, cli_attr, None)
        if cli_val is not None:
            resolved_bool = bool(cli_val)
        elif has_explicit_cfg and (cfg_key in tool_cfg or calib_key in tool_cfg):
            resolved_bool = bool(tool_cfg.get(cfg_key, tool_cfg.get(calib_key)))
        elif calib_key in active_calib:
            resolved_bool = _safe_bool(active_calib[calib_key])
        else:
            resolved_bool = bool(tool_cfg.get(cfg_key, tool_cfg.get(calib_key, default_val)))
        if calib_key == "filter_stop_shingles" and getattr(args, "diff_only", False) and cli_val is None:
            resolved_bool = True
        setattr(args, cli_attr, resolved_bool)

    for extra_flag in ("nms", "merge_subtrees", "tfidf", "gapped_tolerance"):
        extra_val = getattr(args, extra_flag, None)
        setattr(args, extra_flag, bool(tool_cfg.get(extra_flag, False)) if extra_val is None else bool(extra_val))

    call_seq_enabled = bool(args.call_sequences)
    audit_tests_enabled = bool(args.audit_tests)
    idioms_enabled = bool(args.idioms)
    stop_shingles_enabled = bool(args.stop_shingles)

    if args.type4 and _run_type4_semantic_audit(target, excludes, args.strict_type4, use_color):
        return 1

    mode_desc = _format_scan_mode_description(
        args,
        strip_annotations=strip_annotations,
        strip_docstrings=strip_docstrings,
        idioms_enabled=idioms_enabled,
        call_seq_enabled=call_seq_enabled,
        audit_tests_enabled=audit_tests_enabled,
        stop_shingles_enabled=stop_shingles_enabled,
    )

    if args.format == "text":
        print(f"\nScanning '{target}' for AST structural clones (threshold >= {threshold:.0%}, min_lines={min_lines}, mode: {mode_desc})...")

    sort_by = "priority" if args.priority else args.sort_by

    if calib_dict and isinstance(calib_dict, dict) and args.format == "text":
        calib_hash = calib_dict.get("config_hash") or getattr(preloaded_baseline, "config_hash", None)
        if calib_hash:
            active_cfg: Dict[str, Any] = {
                "bag_of_tokens": args.bag_of_tokens,
                "call_sequences": call_seq_enabled,
                "filter_stop_shingles": stop_shingles_enabled,
                "audit_tests": bool(audit_tests_enabled),
                "include_notebooks": bool(args.notebooks),
                "max_index_frequency": max_index_frequency,
                "min_corpus_size": min_corpus_size,
                "min_lines": min_lines,
                "min_tokens": min_tokens,
                "functions_only": args.functions_only,
                "sliding_window": args.sliding_window,
                "window_size": effective_window_size,
                "blind_indexing": args.blind_indexing,
                "merge_subtrees": args.merge_subtrees,
                "complex_expressions": args.complex_expressions,
                "min_expr_complexity": effective_min_expr_complexity,
                "clause_level": args.clause_level,
                "data_tables": args.data_tables,
                "strip_annotations": strip_annotations,
                "nms": args.nms,
                "class_level": args.class_level,
                "blind_literals": args.blind_literals,
                "filter_boilerplate": args.filter_boilerplate,
                "consistent_renaming": args.consistent_renaming,
                "harvest_closures": args.harvest_closures,
                "commutative": args.commutative,
                "comprehensions": args.comprehensions,
                "idioms": idioms_enabled,
                "abstract_expressions": args.abstract_expressions,
                "strip_docstrings": strip_docstrings,
                "excludes": excludes,
                "scope": scan_offset,
            }
            active_hash = compute_calibration_config_hash(active_cfg)
            if calib_hash != active_hash:
                print(
                    colorize(
                        f"Warning: Active scan configuration does not match baseline calibration config (baseline: {calib_hash[:8]}, active: {active_hash[:8]}). Calibration shingle frequencies may not align with active scan settings.",
                        COLOR_BOLD + COLOR_YELLOW,
                        use_color,
                    )
                )

    diff_files: Optional[Sequence[str]] = None
    if args.diff_only:
        git_diff_root = git_worktree_root
        raw_diff_files = _safe_call_git_diff_helper(
            get_git_modified_files, args.since, git_diff_root
        )
        if not raw_diff_files:
            mod_ranges = _safe_call_git_diff_helper(
                get_git_modified_line_ranges, args.since, git_diff_root
            )
            if mod_ranges:
                raw_diff_files = list(mod_ranges.keys())
        if raw_diff_files:
            diff_files = _normalize_git_paths_for_target(
                raw_diff_files, git_diff_root, target_repo_root
            )

    scan_res = scan_target(
        target,
        repo_root=target_repo_root,
        diff_files=diff_files,
        min_lines=min_lines,
        min_tokens=min_tokens,
        threshold=threshold,
        excludes=excludes,
        functions_only=args.functions_only,
        sliding_window=args.sliding_window,
        window_size=effective_window_size,
        blind_indexing=args.blind_indexing,
        merge_subtrees=args.merge_subtrees,
        complex_expressions=args.complex_expressions,
        min_expr_complexity=effective_min_expr_complexity,
        clause_level=args.clause_level,
        data_tables=args.data_tables,
        strip_annotations=strip_annotations,
        nms=args.nms,
        class_level=args.class_level,
        blind_literals=args.blind_literals,
        bag_of_tokens=args.bag_of_tokens,
        filter_boilerplate=args.filter_boilerplate,
        consistent_renaming=args.consistent_renaming,
        tfidf=args.tfidf,
        harvest_closures=args.harvest_closures,
        commutative=args.commutative,
        comprehensions=args.comprehensions,
        idioms=idioms_enabled,
        abstract_expressions=args.abstract_expressions,
        gapped_tolerance=args.gapped_tolerance,
        call_sequences=call_seq_enabled,
        audit_tests=audit_tests_enabled,
        exemptions=cfg_exemptions,
        workers=args.workers,
        sort_by=sort_by,
        top_n=args.top,
        include_notebooks=args.notebooks,
        strip_docstrings=strip_docstrings,
        max_index_frequency=max_index_frequency,
        filter_stop_shingles=stop_shingles_enabled,
        min_corpus_size=min_corpus_size,
        corpus_calibration=calib_dict,
        return_calibration=bool(args.record_baseline),
    )

    clones: List[Tuple[float, Dict[str, Any], Dict[str, Any]]]
    recorded_calib: Optional[Dict[str, Any]] = None
    if isinstance(scan_res, tuple) and len(scan_res) == 2 and isinstance(scan_res[1], dict):
        clones, recorded_calib = scan_res
    elif isinstance(scan_res, list):
        clones = scan_res
    else:
        clones = []

    if calib_dict and isinstance(calib_dict, dict) and args.format == "text":
        drift = calib_dict.get("unit_drift")
        if drift is not None and drift >= 0.20:
            curr_units = calib_dict.get("current_units")
            base_units = calib_dict.get("total_units")
            print(
                colorize(
                    f"Warning: Calibration drift detected: repository units drifted by {drift:.1%} from baseline calibration "
                    f"(current: {curr_units}, baseline: {base_units}). "
                    f"Consider re-running with --record-baseline to refresh calibration.",
                    COLOR_BOLD + COLOR_YELLOW,
                    use_color,
                )
            )

    if (calib_dict is not None or preloaded_baseline is not None) and args.format == "text":
        base_commit_raw = (
            (calib_dict.get("recorded_commit") if isinstance(calib_dict, dict) else None)
            or getattr(preloaded_baseline, "recorded_commit", None)
        )
        if base_commit_raw and getattr(args, "verbose", False):
            base_commit = str(base_commit_raw)
            curr_commit = get_git_head_commit(repo_root=git_worktree_root)
            if curr_commit and base_commit.lower() != curr_commit.lower():
                print(
                    colorize(
                        f"Info: Repository HEAD commit {curr_commit[:8]} differs from baseline recorded commit {base_commit[:8]}.",
                        COLOR_YELLOW,
                        use_color,
                    )
                )

    if args.record_baseline:
        try:
            bp = record_baseline(
                clones,
                args.record_baseline,
                target,
                threshold,
                corpus_calibration=recorded_calib,
                repo_root=git_worktree_root,
                clone_basis="target_relative",
            )
        except TypeError:
            bp = record_baseline(
                clones,
                args.record_baseline,
                target,
                threshold,
                corpus_calibration=recorded_calib,
                repo_root=git_worktree_root,
            )
        print(f"[OK] Recorded {len(clones)} clone baseline pair(s) to {bp}")
        return 0

    clones, early_exit = _apply_baseline_and_diff_filters(
        clones,
        args,
        tool_cfg,
        target_repo_root=target_repo_root,
        preloaded_baseline=preloaded_baseline,
        target=target,
    )
    if early_exit is not None:
        return early_exit

    families: Optional[List[Dict[str, Any]]] = None
    if args.cluster:
        linkage_strategy = args.linkage or str(tool_cfg.get("linkage", "single"))
        cluster_floor = (
            args.min_cluster_similarity
            if args.min_cluster_similarity is not None
            else float(tool_cfg.get("min_cluster_similarity", threshold))
        )
        linkage_tol = (
            args.linkage_tolerance
            if args.linkage_tolerance is not None
            else float(tool_cfg.get("linkage_tolerance", 0.0))
        )
        families = cluster_clone_families(
            clones,
            linkage=linkage_strategy,
            min_similarity_floor=cluster_floor,
            linkage_tolerance=linkage_tol,
        )

    stats: Optional[Dict[str, Any]] = None
    if args.stats or args.summary or args.html:
        stats = compute_repository_dry_stats(
            target,
            clones,
            excludes=excludes,
            include_notebooks=getattr(args, "notebooks", False),
        )
        if args.summary:
            _write_artifact_file(args.summary, format_markdown_summary(stats, target), "SUMMARY", args.format == "text")

    cov_data: Dict[str, Set[int]] = {}
    if args.coverage:
        cov_data = read_coverage_data(args.coverage)

    if args.html:
        html_report = generate_html_report(
            clones,
            target,
            threshold,
            families=families,
            stats=stats,
            repo_root=target_repo_root,
        )
        _write_artifact_file(args.html, html_report, "HTML", args.format == "text")

    if args.patch:
        patch_text = (
            generate_refactoring_patch(
                clones,
                repo_root=target_repo_root,
                type_merge_strategy=type_merge_strategy,
                replace_clones=replace_clones,
                method_binding=method_binding,
                cross_file_strategy=cross_file_strategy,
                shared_module_name=shared_module_name,
            )
            if clones
            else ""
        )
        _write_artifact_file(args.patch, patch_text, "PATCH", args.format == "text")
    elif args.replace_clones and args.format == "text":
        print(
            colorize(
                "Warning: --replace-clones specified without --patch; no patch will be generated.",
                COLOR_BOLD + COLOR_YELLOW,
                use_color,
            )
        )

    if args.github_annotations and clones:
        annotations = format_github_annotations(clones)
        if annotations:
            print("\n".join(annotations))

    if args.format in ("sarif", "json"):
        report_data = (
            format_sarif_report(clones, target, threshold)
            if args.format == "sarif"
            else format_json_report(clones, target, threshold, families=families, stats=stats)
        )
        emit_structured_report(
            report_data, "SARIF 2.1.0" if args.format == "sarif" else "JSON", args.output
        )
        return 0 if not clones else 1

    # Text format
    if stats and args.stats:
        _render_dry_scorecard(stats, use_color)

    if not clones:
        ok_msg = colorize(f"[OK] No structural code clones found with similarity >= {threshold:.0%}. Codebase is DRY!", COLOR_BOLD + COLOR_GREEN, use_color)
        print(ok_msg)
        if args.output:
            out_p = Path(args.output)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as fh:
                fh.write(f"[OK] No structural code clones found with similarity >= {threshold:.0%}. Codebase is DRY!\n")
        return 0

    report_lines = _render_text_violations(
        clones,
        threshold,
        sort_by,
        args=args,
        families=families,
        cov_data=cov_data,
        use_color=use_color,
        target_repo_root=target_repo_root,
        audit_tests_enabled=audit_tests_enabled,
    )

    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as fh:
            fh.write("\n".join(report_lines) + "\n")

    return 1


if __name__ == "__main__":
    sys.exit(main())
