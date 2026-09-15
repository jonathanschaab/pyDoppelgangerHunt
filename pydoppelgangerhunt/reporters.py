"""Reporting, formatters (SARIF, JSON, Markdown), unified diffing, and ANSI terminal color."""

from __future__ import annotations

import difflib
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple


# ANSI Color Codes
COLOR_RESET = "\033[0m"
COLOR_BOLD = "\033[1m"
COLOR_RED = "\033[31m"
COLOR_GREEN = "\033[32m"
COLOR_YELLOW = "\033[33m"
COLOR_BLUE = "\033[34m"
COLOR_MAGENTA = "\033[35m"
COLOR_CYAN = "\033[36m"


def supports_color(color_override: Optional[bool] = None) -> bool:
    """Determines whether the terminal environment supports ANSI colors."""
    if color_override is not None:
        return color_override
    if "NO_COLOR" in os.environ:
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def colorize(text: str, color_code: str, enabled: bool) -> str:
    """Wraps text in ANSI color escape codes if color is enabled."""
    if not enabled:
        return text
    return f"{color_code}{text}{COLOR_RESET}"


def extract_unit_source_code(unit: Dict[str, Any], repo_root: Optional[str] = None) -> List[str]:
    """Reads raw source code lines for a given unit from disk."""
    file_path = Path(unit["file"])
    if repo_root and not file_path.is_absolute():
        file_path = Path(repo_root) / file_path
    if not file_path.exists():
        return [f"# Source for {unit['name']} lines {unit['start']}-{unit['end']}\n"]
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
            all_lines = fh.readlines()
        start = max(1, unit["start"])
        end = min(len(all_lines), unit["end"])
        return all_lines[start - 1 : end]
    except OSError:
        return [f"# Unable to read {unit['file']}\n"]


def generate_clone_diff(
    u1: Dict[str, Any],
    u2: Dict[str, Any],
    repo_root: Optional[str] = None,
    color: bool = False,
) -> str:
    """Produces unified line diff between two cloned code blocks, with optional syntax coloring."""
    lines1 = extract_unit_source_code(u1, repo_root)
    lines2 = extract_unit_source_code(u2, repo_root)
    from_label = f"{u1['file']}:{u1['start']}-{u1['end']} ({u1['name']})"
    to_label = f"{u2['file']}:{u2['start']}-{u2['end']} ({u2['name']})"
    diff = list(difflib.unified_diff(
        lines1, lines2, fromfile=from_label, tofile=to_label, lineterm=""
    ))

    if not color:
        return "\n".join(diff)

    colored_diff: List[str] = []
    for line in diff:
        if line.startswith("---") or line.startswith("+++"):
            colored_diff.append(colorize(line, COLOR_BOLD + COLOR_CYAN, True))
        elif line.startswith("@@"):
            colored_diff.append(colorize(line, COLOR_CYAN, True))
        elif line.startswith("-"):
            colored_diff.append(colorize(line, COLOR_RED, True))
        elif line.startswith("+"):
            colored_diff.append(colorize(line, COLOR_GREEN, True))
        else:
            colored_diff.append(line)
    return "\n".join(colored_diff)


def synthesize_refactoring_suggestion(
    u1: Dict[str, Any], u2: Dict[str, Any], repo_root: Optional[str] = None
) -> str:
    """Synthesizes actionable refactoring recommendation and shared helper template."""
    lines1 = extract_unit_source_code(u1, repo_root)
    lines2 = extract_unit_source_code(u2, repo_root)
    n1 = u1["name"].split(":")[-1]
    n2 = u2["name"].split(":")[-1]

    if n1.startswith("test_") and n2.startswith("test_"):
        return (
            f"[REFACTOR SUGGESTION] Combine '{n1}' and '{n2}' into a single parametrized test:\n"
            f"    @pytest.mark.parametrize(\"param1, param2, expected\", [\n"
            f"        (..., ..., ...),  # Test case from {n1}\n"
            f"        (..., ..., ...),  # Test case from {n2}\n"
            f"    ])\n"
            f"    def test_shared_behavior(param1, param2, expected):\n"
            f"        ..."
        )

    shared_prefix = ""
    for c1, c2 in zip(n1, n2):
        if c1 == c2:
            shared_prefix += c1
        else:
            break
    clean_prefix = shared_prefix.strip("_")
    helper_name = f"_shared_{clean_prefix}" if clean_prefix else "_shared_helper"

    diff_lines = [
        line for line in difflib.ndiff(lines1, lines2)
        if line.startswith("- ") or line.startswith("+ ")
    ]
    diff_summary = f"{len(diff_lines)} varying line(s) detected between clone blocks."

    return (
        f"[REFACTOR SUGGESTION] Extract duplicated logic into common helper '{helper_name}':\n"
        f"    # {diff_summary}\n"
        f"    def {helper_name}(*args, **kwargs):\n"
        f"        \"\"\"Extracted common implementation for {n1} and {n2}.\"\"\"\n"
        f"        # Place common block here and parameterize differing inputs."
    )


def format_sarif_report(
    clones: List[Tuple[float, Dict[str, Any], Dict[str, Any]]],
    target: str,
    threshold: float,
) -> Dict[str, Any]:
    """Formats detected clones into OASIS SARIF 2.1.0 standard schema for GitHub Code Scanning."""
    results: List[Dict[str, Any]] = []
    for idx, (sim, u1, u2) in enumerate(clones):
        rule_id = "PYDOPPEL001"
        message = (
            f"Code duplication: AST block '{u1['name']}' in {u1['file']}:{u1['start']}-{u1['end']} "
            f"is {sim:.1%} structurally identical to '{u2['name']}' in {u2['file']}:{u2['start']}-{u2['end']} "
            f"(threshold >= {threshold:.0%})."
        )
        f1_norm = u1["file"].replace("\\", "/")
        f2_norm = u2["file"].replace("\\", "/")
        result_item: Dict[str, Any] = {
            "ruleId": rule_id,
            "ruleIndex": 0,
            "level": "warning" if sim < 0.95 else "error",
            "message": {"text": message},
            "locations": [
                {
                    "physicalLocation": {
                        "artifactLocation": {
                            "uri": f1_norm,
                            "uriBaseId": "%SRCROOT%",
                        },
                        "region": {
                            "startLine": u1["start"],
                            "endLine": u1["end"],
                        },
                    }
                }
            ],
            "relatedLocations": [
                {
                    "id": idx + 1,
                    "message": {"text": f"Clone twin: '{u2['name']}' in {f2_norm}:{u2['start']}-{u2['end']}"},
                    "physicalLocation": {
                        "artifactLocation": {
                            "uri": f2_norm,
                            "uriBaseId": "%SRCROOT%",
                        },
                        "region": {
                            "startLine": u2["start"],
                            "endLine": u2["end"],
                        },
                    },
                }
            ],
            "properties": {
                "similarity": round(sim, 4),
                "cloneKind": u1.get("kind", "function"),
            },
        }
        results.append(result_item)

    return {
        "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "pyDoppelgangerHunt",
                        "version": "1.0.0",
                        "informationUri": "https://github.com/jonathanschaab/pyDoppelgangerHunt",
                        "rules": [
                            {
                                "id": "PYDOPPEL001",
                                "name": "StructuralCodeClone",
                                "shortDescription": {"text": "Structural code duplication detected across AST units."},
                                "fullDescription": {
                                    "text": (
                                        "A structural AST code clone was detected that violates DRY principles. "
                                        "Refactor the common logic into a shared helper function or base model."
                                    )
                                },
                                "defaultConfiguration": {"level": "warning"},
                            }
                        ],
                    }
                },
                "results": results,
                "properties": {
                    "target": target,
                    "threshold": threshold,
                    "cloneCount": len(clones),
                },
            }
        ],
    }


def format_json_report(
    clones: List[Tuple[float, Dict[str, Any], Dict[str, Any]]],
    target: str,
    threshold: float,
    families: Optional[List[Dict[str, Any]]] = None,
    stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generates structured JSON representation of clone matches."""
    data: Dict[str, Any] = {
        "tool": "pyDoppelgangerHunt",
        "version": "1.0.0",
        "target": target,
        "threshold": threshold,
        "clone_count": len(clones),
        "clones": [
            {
                "similarity": round(sim, 4),
                "unit_a": {
                    "name": u1["name"],
                    "file": u1["file"].replace("\\", "/"),
                    "start": u1["start"],
                    "end": u1["end"],
                    "kind": u1.get("kind"),
                    "tokens": u1.get("token_count", 0),
                },
                "unit_b": {
                    "name": u2["name"],
                    "file": u2["file"].replace("\\", "/"),
                    "start": u2["start"],
                    "end": u2["end"],
                    "kind": u2.get("kind"),
                    "tokens": u2.get("token_count", 0),
                },
            }
            for sim, u1, u2 in clones
        ],
    }
    if families is not None:
        data["families"] = [
            {
                "family_id": f["family_id"],
                "member_count": f["member_count"],
                "unique_files": f["unique_files"],
                "avg_similarity": round(f["avg_similarity"], 4),
                "max_similarity": round(f["max_similarity"], 4),
                "total_lines": f["total_lines"],
                "members": [
                    {
                        "name": m["name"],
                        "file": m["file"].replace("\\", "/"),
                        "start": m["start"],
                        "end": m["end"],
                        "kind": m.get("kind"),
                    }
                    for m in f["members"]
                ],
            }
            for f in families
        ]
    if stats is not None:
        data["stats"] = stats
    return data


def format_markdown_summary(stats: Dict[str, Any], target: str) -> str:
    """Formats GitHub Step Summary Markdown report."""
    md_lines: List[str] = [
        "## pyDoppelgangerHunt DRY Quality Audit Summary",
        "",
        f"**Audit Target:** `{target}`",
        "",
        "| Metric | Value |",
        "| :--- | :--- |",
        f"| **Repository DRY Score** | **{stats['dry_score']:.1f}% (Grade: {stats['grade']})** |",
        f"| **Total Source Lines (SLOC)** | {stats['sloc']:,} |",
        f"| **Duplicated Lines (DLOC)** | {stats['dloc']:,} |",
        f"| **Duplication Rate** | {stats['duplication_pct']:.2f}% |",
        f"| **Total Clone Pairs** | {stats['clone_pairs']} |",
        f"| **Clone Families** | {stats['clone_families']} |",
        "",
    ]
    if stats.get("package_sloc"):
        md_lines.append("### Package SLOC Breakdown")
        md_lines.append("| Package / Directory | SLOC |")
        md_lines.append("| :--- | :--- |")
        for pkg, lines in sorted(stats["package_sloc"].items(), key=lambda x: x[1], reverse=True):
            md_lines.append(f"| `{pkg}` | {lines:,} |")
        md_lines.append("")

    return "\n".join(md_lines)


def emit_structured_report(
    data: Dict[str, Any], fmt_label: str, output_path: Optional[str]
) -> None:
    """Emits formatted JSON or SARIF report to stdout or output file."""
    text = json.dumps(data, indent=2)
    if output_path:
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"Report saved in {fmt_label} format to {output_path}")
    else:
        print(text)


def format_github_annotations(
    clones: List[Tuple[float, Dict[str, Any], Dict[str, Any]]]
) -> List[str]:
    """Generates GitHub Actions workflow commands to annotate PR line diffs."""
    annotations: List[str] = []
    for sim, u1, u2 in clones:
        f1 = u1["file"].replace("\\", "/").split("#")[0]
        s1 = u1["start"]
        e1 = u1["end"]
        f2 = u2["file"].replace("\\", "/").split("#")[0]
        s2 = u2["start"]
        e2 = u2["end"]
        msg1 = f"Structural clone ({sim:.1%}) matching {f2}:{s2}-{e2} ({u2['name']})"
        annotations.append(
            f"::warning file={f1},line={s1},endLine={e1},title=pyDoppelgangerHunt Duplicate Code::{msg1}"
        )
        msg2 = f"Structural clone ({sim:.1%}) matching {f1}:{s1}-{e1} ({u1['name']})"
        annotations.append(
            f"::warning file={f2},line={s2},endLine={e2},title=pyDoppelgangerHunt Duplicate Code::{msg2}"
        )
    return annotations


def generate_html_report(
    clones: List[Tuple[float, Dict[str, Any], Dict[str, Any]]],
    target: str,
    threshold: float,
    families: Optional[List[Dict[str, Any]]] = None,
    stats: Optional[Dict[str, Any]] = None,
) -> str:
    """Generates a standalone, self-contained interactive HTML audit report."""
    dry_score = stats["dry_score"] if stats else 100.0
    grade = stats["grade"] if stats else "A+"
    sloc = stats["sloc"] if stats else 0
    dloc = stats["dloc"] if stats else 0
    duplication_pct = stats["duplication_pct"] if stats else 0.0

    meter_color = "#22c55e" if dry_score >= 90 else ("#eab308" if dry_score >= 80 else "#ef4444")
    circumference = 282.74  # 2 * pi * 45
    dashoffset = circumference * (1.0 - (dry_score / 100.0))

    cards_html: List[str] = []
    for idx, (sim, u1, u2) in enumerate(clones, 1):
        diff_txt = generate_clone_diff(u1, u2)
        sug_txt = synthesize_refactoring_suggestion(u1, u2)
        diff_block = (
            f"<pre class='diff'><code>{diff_txt}</code></pre>" if diff_txt else "<em>No textual diff</em>"
        )
        sug_block = (
            f"<div class='sug'><strong>Refactoring Suggestion:</strong><pre><code>{sug_txt}</code></pre></div>"
            if sug_txt
            else ""
        )

        card = f"""
        <div class="clone-card" data-search="{u1['file']} {u1['name']} {u2['file']} {u2['name']}">
            <div class="card-header">
                <span class="sim-badge" style="background: {'#ef4444' if sim >= 0.85 else '#f59e0b'};">
                    {sim:.1%} Match
                </span>
                <span class="card-title">#{idx}: {u1['name']} &harr; {u2['name']}</span>
            </div>
            <div class="card-body">
                <div class="loc-row">
                    <span class="file-tag">&#128196; {u1['file']}:{u1['start']}-{u1['end']}</span>
                    <span class="arrow">&harr;</span>
                    <span class="file-tag">&#128196; {u2['file']}:{u2['start']}-{u2['end']}</span>
                </div>
                {sug_block}
                <details>
                    <summary>View Unified AST Diff</summary>
                    {diff_block}
                </details>
            </div>
        </div>
        """
        cards_html.append(card)

    families_html: List[str] = []
    if families:
        for fam in families:
            members_li = "".join(
                f"<li><code>{m['file']}:{m['start']}-{m['end']}</code> ({m['name']})</li>"
                for m in fam["members"]
            )
            f_card = f"""
            <div class="family-card">
                <h3>{fam['family_id']} &mdash; {fam['member_count']} Members ({fam['avg_similarity']:.1%} avg sim)</h3>
                <ul>{members_li}</ul>
            </div>
            """
            families_html.append(f_card)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>pyDoppelgangerHunt Audit Report - {target}</title>
    <style>
        :root {{
            --bg: #0f172a;
            --surface: #1e293b;
            --border: #334155;
            --text: #f8fafc;
            --muted: #94a3b8;
            --primary: #38bdf8;
            --success: #22c55e;
            --warning: #f59e0b;
            --danger: #ef4444;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background: var(--bg);
            color: var(--text);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            padding: 2rem;
            line-height: 1.5;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        header {{ margin-bottom: 2rem; border-bottom: 1px solid var(--border); padding-bottom: 1.5rem; }}
        h1 {{ font-size: 1.8rem; color: var(--primary); margin-bottom: 0.5rem; }}
        .meta {{ color: var(--muted); font-size: 0.95rem; }}
        .grid-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        .stat-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1.25rem;
            text-align: center;
        }}
        .stat-card .val {{ font-size: 1.8rem; font-weight: bold; margin-top: 0.5rem; color: var(--text); }}
        .stat-card .lbl {{ font-size: 0.85rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }}
        .gauge-container {{ display: flex; align-items: center; justify-content: center; position: relative; }}
        .gauge-text {{
            position: absolute;
            font-size: 1.4rem;
            font-weight: bold;
            color: {meter_color};
            text-align: center;
        }}
        .search-box {{
            width: 100%;
            padding: 0.75rem 1rem;
            border-radius: 6px;
            border: 1px solid var(--border);
            background: var(--surface);
            color: var(--text);
            font-size: 1rem;
            margin-bottom: 1.5rem;
        }}
        .clone-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            margin-bottom: 1rem;
            overflow: hidden;
        }}
        .card-header {{
            padding: 1rem 1.25rem;
            background: rgba(255, 255, 255, 0.03);
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            gap: 1rem;
        }}
        .sim-badge {{
            padding: 0.25rem 0.6rem;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: bold;
            color: #fff;
        }}
        .card-title {{ font-family: monospace; font-weight: 600; }}
        .card-body {{ padding: 1.25rem; }}
        .loc-row {{ display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1rem; flex-wrap: wrap; }}
        .file-tag {{
            background: rgba(56, 189, 248, 0.1);
            color: var(--primary);
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-family: monospace;
            font-size: 0.9rem;
        }}
        .arrow {{ color: var(--muted); }}
        .sug {{
            background: rgba(245, 158, 11, 0.08);
            border-left: 3px solid var(--warning);
            padding: 0.75rem 1rem;
            border-radius: 4px;
            margin-bottom: 1rem;
            font-size: 0.9rem;
        }}
        details {{ margin-top: 0.5rem; }}
        summary {{ cursor: pointer; color: var(--primary); font-size: 0.9rem; margin-bottom: 0.5rem; }}
        pre.diff {{
            background: #090d16;
            padding: 1rem;
            border-radius: 6px;
            overflow-x: auto;
            font-size: 0.85rem;
            color: #cbd5e1;
            font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
        }}
        .family-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1.25rem;
            margin-bottom: 1rem;
        }}
        .family-card h3 {{ font-size: 1.1rem; color: var(--primary); margin-bottom: 0.5rem; }}
        .family-card ul {{ margin-left: 1.5rem; color: var(--muted); font-size: 0.9rem; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>pyDoppelgangerHunt Audit Report</h1>
            <div class="meta">Target: <code>{target}</code> | Similarity Threshold: {threshold:.0%} | Generated by pyDoppelgangerHunt v1.0.0</div>
        </header>

        <div class="grid-stats">
            <div class="stat-card">
                <div class="gauge-container">
                    <svg width="120" height="120" viewBox="0 0 100 100">
                        <circle cx="50" cy="50" r="45" stroke="#334155" stroke-width="8" fill="none" />
                        <circle cx="50" cy="50" r="45" stroke="{meter_color}" stroke-width="8" fill="none"
                                stroke-dasharray="{circumference}" stroke-dashoffset="{dashoffset}"
                                stroke-linecap="round" transform="rotate(-90 50 50)" />
                    </svg>
                    <div class="gauge-text">{dry_score:.1f}%<br><span style="font-size: 0.9rem; font-weight: normal;">Grade {grade}</span></div>
                </div>
                <div class="lbl" style="margin-top: 0.5rem;">DRY Score</div>
            </div>
            <div class="stat-card">
                <div class="lbl">Source Lines (SLOC)</div>
                <div class="val">{sloc:,}</div>
            </div>
            <div class="stat-card">
                <div class="lbl">Duplicated Lines (DLOC)</div>
                <div class="val">{dloc:,}</div>
            </div>
            <div class="stat-card">
                <div class="lbl">Duplication Rate</div>
                <div class="val">{duplication_pct:.2f}%</div>
            </div>
            <div class="stat-card">
                <div class="lbl">Clone Pairs</div>
                <div class="val">{len(clones)}</div>
            </div>
        </div>

        {"<h2>Clone Families</h2>" if families else ""}
        {"".join(families_html)}

        <h2 style="margin-top: 2rem; margin-bottom: 1rem;">Detected Duplicate Code Clones</h2>
        <input type="text" id="searchInput" class="search-box" placeholder="Filter clones by path, function, or keyword..." onkeyup="filterCards()">

        <div id="cardsList">
            {"".join(cards_html) if cards_html else "<p style='color: var(--muted);'>No code clones detected with similarity &ge; " + f"{threshold:.0%}. Codebase is DRY!</p>"}
        </div>
    </div>

    <script>
        function filterCards() {{
            const filter = document.getElementById('searchInput').value.toLowerCase();
            const cards = document.querySelectorAll('.clone-card');
            cards.forEach(card => {{
                const searchData = card.getAttribute('data-search').toLowerCase();
                if (searchData.includes(filter)) {{
                    card.style.display = '';
                }} else {{
                    card.style.display = 'none';
                }}
            }});
        }}
    </script>
</body>
</html>"""
    return html_content
