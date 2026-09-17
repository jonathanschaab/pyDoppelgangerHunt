"""Command Line Interface (CLI) entrypoint for pyDoppelgangerHunt."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Set

from pydoppelgangerhunt.baseline import (
    filter_clones_by_baseline,
    load_baseline,
    prune_baseline,
    record_baseline,
)
from pydoppelgangerhunt.clustering import cluster_clone_families
from pydoppelgangerhunt.config import (
    DEFAULT_EXCLUDES,
    init_tool_configuration,
    load_tool_config,
    normalize_path_string,
)
from pydoppelgangerhunt.coverage import check_asymmetric_coverage, read_coverage_data
from pydoppelgangerhunt.fixer import generate_refactoring_patch
from pydoppelgangerhunt.git_diff import (
    check_temporal_divergence,
    filter_clones_by_git_diff,
    get_git_modified_line_ranges,
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
    parser.add_argument("--window-size", type=int, default=5, help="Statement window size for sliding window scanner (default: 5)")
    parser.add_argument("--min-expr-complexity", type=int, default=4, help="Minimum operator complexity for complex expression detection (default: 4)")
    parser.add_argument("--config", type=str, default=None, help="Path to TOML configuration file (defaults to pyproject.toml)")
    parser.add_argument("--format", choices=["text", "json", "sarif"], default="text", help="Output report format (default: text)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output file path to save report")
    parser.add_argument("--html", type=str, default=None, help="Path to write standalone interactive HTML report")
    parser.add_argument("--patch", type=str, default=None, help="Path to write git-apply compatible refactoring patch file")
    parser.add_argument("--sort-by", choices=["similarity", "priority", "sloc"], default="similarity", help="Sort clones by similarity, priority, or sloc (default: similarity)")
    parser.add_argument("--priority", action="store_true", help="Shortcut to sort clones by Priority score (Similarity * SLOC * Complexity)")
    parser.add_argument("--top", type=int, default=None, help="Truncate report to top N clone pairs")
    parser.add_argument("--blame", action="store_true", help="Audit git commit history and warn on temporally divergent clones (>90 days)")
    parser.add_argument("--coverage", type=str, default=None, help="Path to .coverage (SQLite) or coverage.xml to detect asymmetric test coverage")
    parser.add_argument("--notebooks", action="store_true", help="Include Jupyter Notebook (.ipynb) code cells in clone scan")
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
        ("--preserve-annotations", "Preserve PEP 484/526 type annotations during AST shingling"),
        ("--preserve-docstrings", "Preserve docstrings during AST token extraction (docstrings stripped by default)"),
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
        ("--strict-type4", "Fail with non-zero exit code if Type-4 semantic issues are found"),
        ("--stop-shingles", "Filter canonical boilerplate shingles (logging, main guards) to reduce spurious candidate pairs"),
    ]
    for flag_name, help_text in bool_flags:
        parser.add_argument(flag_name, action="store_true", help=help_text)
    parser.add_argument("--type4", "--semantic", action="store_true", help="Also run Type-4 semantic clone & consistency audit")

    return parser


def _write_artifact_file(dest_path: str, content: str, label: str, verbose: bool = True) -> None:
    """Writes report content to disk and emits console confirmation."""
    with open(dest_path, "w", encoding="utf-8") as fh:
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
    tool_cfg = load_tool_config(args.config)
    cfg_target = str(tool_cfg.get("target") or "")
    default_dir = (
        cfg_target
        if cfg_target and os.path.isdir(cfg_target)
        else (
            "pyfloorplanner"
            if os.path.isdir("pyfloorplanner")
            else ("pydoppelgangerhunt" if os.path.isdir("pydoppelgangerhunt") else ".")
        )
    )
    target = target_arg or default_dir
    target_repo_root = target if os.path.isdir(target) else (os.path.dirname(target) or ".")
    threshold = args.threshold if args.threshold is not None else float(tool_cfg.get("threshold", 0.90))
    min_lines = args.min_lines if args.min_lines is not None else int(tool_cfg.get("min_lines", 8))
    min_tokens = args.min_tokens if args.min_tokens is not None else int(tool_cfg.get("min_tokens", 15))

    cfg_excludes = list(tool_cfg.get("exclude", DEFAULT_EXCLUDES))
    excludes = cfg_excludes + args.exclude

    cfg_exemptions = (
        [tuple(p) for p in tool_cfg["exemptions"]] if "exemptions" in tool_cfg else None
    )

    call_seq_enabled = args.call_sequences or bool(tool_cfg.get("call_sequences", False))
    audit_tests_enabled = args.audit_tests or bool(tool_cfg.get("audit_tests", False))
    idioms_enabled = args.idioms or bool(tool_cfg.get("idioms", False))
    stop_shingles_enabled = (
        args.stop_shingles
        or args.diff_only
        or bool(tool_cfg.get("stop_shingles", False))
    )

    max_index_frequency = (
        args.max_index_frequency
        if args.max_index_frequency is not None
        else float(tool_cfg.get("max_index_frequency", 0.25))
    )
    min_corpus_size = (
        args.min_corpus_units
        if args.min_corpus_units is not None
        else (int(tool_cfg["min_corpus_units"]) if "min_corpus_units" in tool_cfg else None)
    )
    strip_docstrings = not args.preserve_docstrings
    strip_annotations = True if args.strip_annotations else (not args.preserve_annotations)

    raw_binding = args.method_binding or str(tool_cfg.get("method_binding", "auto"))
    method_binding = raw_binding if raw_binding in ("auto", "method", "module") else "auto"

    if args.type4:
        try:
            import importlib  # pylint: disable=import-outside-toplevel
            mod = importlib.import_module("check_semantic_clones")
            find_semantic_clones = getattr(mod, "find_semantic_clones")
            report_semantic_results = getattr(mod, "report_semantic_results")
            print(f"Scanning '{target}' for Type-4 Semantic Clones & Consistency Violations...")
            results = find_semantic_clones(target, excludes=excludes)
            has_violations = report_semantic_results(results)
            if has_violations and args.strict_type4:
                print(colorize("[FAIL] Type-4 semantic clone violations detected!", COLOR_BOLD + COLOR_RED, use_color))
                return 1
        except (ImportError, AttributeError):
            print("[INFO] check_semantic_clones not found; skipping Type-4 scan.")

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
    mode_desc = " + ".join(modes)

    if args.format == "text":
        print(f"\nScanning '{target}' for AST structural clones (threshold >= {threshold:.0%}, min_lines={min_lines}, mode: {mode_desc})...")

    sort_by = "priority" if args.priority else args.sort_by

    clones = scan_target(
        target,
        min_lines=min_lines,
        min_tokens=min_tokens,
        threshold=threshold,
        excludes=excludes,
        functions_only=args.functions_only,
        sliding_window=args.sliding_window,
        window_size=args.window_size,
        blind_indexing=args.blind_indexing,
        merge_subtrees=args.merge_subtrees,
        complex_expressions=args.complex_expressions,
        min_expr_complexity=args.min_expr_complexity,
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
    )

    if args.record_baseline:
        bp = record_baseline(clones, args.record_baseline, target, threshold)
        print(f"[OK] Recorded {len(clones)} clone baseline pair(s) to {bp}")
        return 0

    baseline_path = args.baseline or tool_cfg.get("baseline")

    if args.prune_baseline:
        if not baseline_path:
            print("[ERROR] --prune-baseline requires a baseline path (specify via --baseline or config)")
            return 1
        if not os.path.exists(baseline_path):
            print(f"[ERROR] Baseline file '{baseline_path}' not found")
            return 1
        try:
            prune_res = prune_baseline(
                baseline_path, clones, repo_root=target_repo_root
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
        base_fps = load_baseline(baseline_path)
        clones, suppressed_count = filter_clones_by_baseline(clones, base_fps)
        if args.format == "text":
            print(f"[BASELINE] Suppressed {suppressed_count} grandfathered clone(s). {len(clones)} un-grandfathered clone(s) remaining.")

    if args.diff_only:
        diff_policy = args.partial_hunk_policy or str(tool_cfg.get("partial_hunk_policy", "any"))
        min_overlap = (
            args.min_diff_overlap
            if args.min_diff_overlap is not None
            else float(tool_cfg.get("min_diff_overlap", 0.0))
        )
        try:
            modified_ranges = get_git_modified_line_ranges(
                since_ref=args.since, repo_root=target_repo_root
            )
        except TypeError:
            modified_ranges = get_git_modified_line_ranges(since_ref=args.since)
        clones = filter_clones_by_git_diff(
            clones,
            modified_ranges,
            policy=diff_policy,
            min_overlap_ratio=min_overlap,
        )
        if args.format == "text":
            print(f"[INFO] Filtered by git diff (policy={diff_policy}): {len(clones)} clone pair(s) touch modified lines.")

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
                method_binding=method_binding,
            )
            if clones
            else ""
        )
        _write_artifact_file(args.patch, patch_text, "PATCH", args.format == "text")

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
        print("\n" + "=" * 55)
        grade_color = COLOR_GREEN if stats["grade"] in ("A+", "A") else (COLOR_YELLOW if stats["grade"] in ("B", "C") else COLOR_RED)
        score_str = colorize(f"{stats['dry_score']:.1f}% (Grade: {stats['grade']})", COLOR_BOLD + grade_color, use_color)
        print(f"pyDoppelgangerHunt DRY Scorecard: {score_str}")
        print(f"SLOC: {stats['sloc']:,} | DLOC: {stats['dloc']:,} | Duplication: {stats['duplication_pct']:.2f}%")
        print(f"Clone Pairs: {stats['clone_pairs']} | Clone Families: {stats['clone_families']}")
        print("=" * 55)

    if not clones:
        ok_msg = colorize(f"[OK] No structural code clones found with similarity >= {threshold:.0%}. Codebase is DRY!", COLOR_BOLD + COLOR_GREEN, use_color)
        print(ok_msg)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as fh:
                fh.write(f"[OK] No structural code clones found with similarity >= {threshold:.0%}. Codebase is DRY!\n")
        return 0

    report_lines: List[str] = []
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
                        cov_data=cov_data,
                        use_color=use_color,
                        indent="      ",
                        repo_root=target_repo_root,
                    )
                )
            if args.suggest and len(fam["members"]) >= 2:
                sug = synthesize_refactoring_suggestion(
                    fam["members"][0],
                    fam["members"][1],
                    repo_root=target_repo_root,
                )
                sug_colored = colorize(sug, COLOR_YELLOW, use_color)
                print("      " + sug_colored.replace("\n", "\n      "))
                report_lines.append("      " + sug.replace("\n", "\n      "))
            if args.diff and len(fam["members"]) >= 2:
                diff_out = generate_clone_diff(
                    fam["members"][0],
                    fam["members"][1],
                    repo_root=target_repo_root,
                    color=use_color,
                )
                if diff_out:
                    print("      --- Diff ---")
                    print("      " + diff_out.replace("\n", "\n      "))
                    report_lines.append("      --- Diff ---\n      " + diff_out.replace("\n", "\n      "))
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
                    cov_data=cov_data,
                    use_color=use_color,
                    indent="    ",
                    repo_root=target_repo_root,
                )
            )
            if audit_tests_enabled and n1.startswith("test_") and n2.startswith("test_"):
                tip = colorize("    [TIP] Consider refactoring with @pytest.mark.parametrize", COLOR_YELLOW, use_color)
                print(tip)
                report_lines.append("    [TIP] Consider refactoring with @pytest.mark.parametrize")
            if args.suggest:
                sug = synthesize_refactoring_suggestion(
                    u1, u2, repo_root=target_repo_root
                )
                sug_colored = colorize(sug, COLOR_YELLOW, use_color)
                print("    " + sug_colored.replace("\n", "\n    "))
                report_lines.append("    " + sug.replace("\n", "\n    "))
            if args.diff:
                diff_out = generate_clone_diff(
                    u1, u2, repo_root=target_repo_root, color=use_color
                )
                if diff_out:
                    print("    --- Diff ---")
                    print("    " + diff_out.replace("\n", "\n    "))
                    report_lines.append("    --- Diff ---\n    " + diff_out.replace("\n", "\n    "))

    footer = "\nPlease refactor structural duplicates into shared helpers, base models, or declarative specifications."
    print(footer)
    report_lines.append(footer)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write("\n".join(report_lines) + "\n")

    return 1


if __name__ == "__main__":
    sys.exit(main())
