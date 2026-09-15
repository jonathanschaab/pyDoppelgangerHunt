#!/usr/bin/env python3
"""Quality Gates Runner for pyDoppelgangerHunt.

Reads quality gate specifications from gates.toml and executes them sequentially.
Serves as the single source of truth for CI workflows and local runner scripts.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple


def _load_gates_toml(path: Path) -> Dict[str, Any]:
    """Parse gates.toml using tomllib/tomli or fallback line parser."""
    if not path.is_file():
        raise FileNotFoundError(f"Gate specification not found: {path}")

    # 1. Try tomllib (Python 3.11+)
    try:
        import tomllib  # type: ignore[import-not-found]
        with open(path, "rb") as f:
            return tomllib.load(f)
    except ModuleNotFoundError:
        pass

    # 2. Try tomli (Python < 3.11)
    try:
        import tomli  # type: ignore[import-not-found,import-untyped]
        with open(path, "rb") as f:
            return tomli.load(f)
    except ModuleNotFoundError:
        pass

    # 3. Robust fallback line-by-line parser
    data: Dict[str, Any] = {"gates": {}}
    current_section: Optional[str] = None
    lines = path.read_text(encoding="utf-8").splitlines()

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        section_match = re.match(r"^\[([a-zA-Z0-9_\.]+)\]$", line)
        if section_match:
            current_section = section_match.group(1)
            parts = current_section.split(".")
            cursor = data
            for part in parts:
                if part not in cursor:
                    cursor[part] = {}
                cursor = cursor[part]
            continue

        if current_section and "=" in line:
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip()

            parsed_val: Any = val.strip('"\'')
            if val.lower() == "true":
                parsed_val = True
            elif val.lower() == "false":
                parsed_val = False
            elif val.startswith("[") and val.endswith("]"):
                items = val[1:-1].split(",")
                parsed_val = [item.strip().strip('"\'') for item in items if item.strip()]

            # Navigate to current section dict
            parts = current_section.split(".")
            cursor = data
            for part in parts:
                cursor = cursor[part]
            cursor[key] = parsed_val

    return data


def _colorize(text: str, color_code: str) -> str:
    """Format text with ANSI escape codes if terminal supports it."""
    if os.environ.get("NO_COLOR") or not sys.stdout.isatty():
        return text
    return f"\033[{color_code}m{text}\033[0m"


def _cyan(text: str) -> str:
    return _colorize(text, "0;36")


def _green(text: str) -> str:
    return _colorize(text, "0;32")


def _red(text: str) -> str:
    return _colorize(text, "0;31")


def _yellow(text: str) -> str:
    return _colorize(text, "1;33")


def _parse_version(v_str: str) -> Tuple[int, ...]:
    """Parse version string into tuple of ints for comparison."""
    return tuple(int(x) for x in v_str.split("."))


def run_gates(
    spec_path: Path,
    target_gate: Optional[str] = None,
    list_only: bool = False,
    fast: bool = False,
) -> int:
    """Execute quality gates defined in gates.toml specification."""
    try:
        spec = _load_gates_toml(spec_path)
    except Exception as exc:
        print(_red(f"Error loading {spec_path}: {exc}"), file=sys.stderr)
        return 1

    gates_dict: Dict[str, Any] = spec.get("gates", {})
    if not gates_dict:
        print(_red("No gates found in specification."), file=sys.stderr)
        return 1

    sorted_gate_keys = sorted(gates_dict.keys(), key=lambda k: int(k) if k.isdigit() else k)

    if list_only:
        print(_cyan("\nAvailable Quality Gates:"))
        print(_cyan("=" * 60))
        for k in sorted_gate_keys:
            g = gates_dict[k]
            gate_id = g.get("id", f"gate-{k}")
            name = g.get("name", "Unnamed Gate")
            desc = g.get("description", "")
            print(f" Gate {k} [{gate_id}]: {name}")
            if desc:
                print(f"   Description: {desc}")
            cmd = g.get("command") or g.get("commands", [])
            print(f"   Command: {cmd}\n")
        return 0

    py_ver = sys.version_info[:2]
    py_ver_str = f"{py_ver[0]}.{py_ver[1]}"
    python_exe = sys.executable

    print(_cyan("\n========================================================"))
    print(_cyan(" pyDoppelgangerHunt Quality Gates (gates.toml)"))
    print(_cyan(f" Python {py_ver_str} | Single Source of Truth Runner"))
    print(_cyan("========================================================"))

    start_all = time.time()
    passed_count = 0
    skipped_count = 0
    total_selected = 0

    for k in sorted_gate_keys:
        g = gates_dict[k]
        gate_num = str(k)
        gate_id = g.get("id", f"gate-{gate_num}")
        name = g.get("name", f"Gate {gate_num}")
        min_python = g.get("min_python")

        if target_gate and target_gate not in (gate_num, gate_id):
            continue

        total_selected += 1

        print(f"\n{_cyan('========================================================')}")
        print(_cyan(f" [GATE {gate_num}] {name}"))
        print(_cyan("========================================================"))

        # Check Python version constraint
        if min_python:
            req_ver = _parse_version(str(min_python))
            if py_ver < req_ver:
                print(_yellow(f"[SKIPPED] Gate {gate_num}: {name} (Requires Python >={min_python}, current: {py_ver_str})\n"))
                skipped_count += 1
                continue

        raw_commands = g.get("commands") or [g.get("command", "")]
        commands = [raw_commands] if isinstance(raw_commands, str) else list(raw_commands)

        gate_failed = False
        for cmd in commands:
            if not cmd:
                continue

            # Replace 'python ' with current python interpreter executable
            if cmd.startswith("python "):
                exec_cmd = f'"{python_exe}" ' + cmd[7:]
            elif cmd.startswith("python.exe "):
                exec_cmd = f'"{python_exe}" ' + cmd[11:]
            else:
                exec_cmd = cmd

            proc = subprocess.run(exec_cmd, shell=True, check=False)
            if proc.returncode != 0:
                print(_red(f"\n[FAILED] Gate {gate_num}: {name}"))
                print(_red(f"Command failed with exit code {proc.returncode}: {cmd}\n"))
                gate_failed = True
                break

        if gate_failed:
            return 1

        print(_green(f"[PASSED] Gate {gate_num}: {name}\n"))
        passed_count += 1

    elapsed = time.time() - start_all
    print(_green("========================================================"))
    print(_green(f" ALL {passed_count} EXECUTED GATES PASSED CLEANLY in {elapsed:.1f}s! ({skipped_count} skipped)"))
    print(_green("========================================================\n"))
    return 0


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="pyDoppelgangerHunt Quality Gates Runner")
    parser.add_argument("--gate", "-g", default=None, help="Run specific gate by ID or number")
    parser.add_argument("--list", "-l", action="store_true", help="List all available gates")
    parser.add_argument("--fast", action="store_true", help="Fast execution mode")
    parser.add_argument(
        "--spec",
        default="gates.toml",
        help="Path to gates specification file (default: gates.toml)",
    )
    args = parser.parse_args()

    spec_path = Path(args.spec)
    if not spec_path.is_absolute():
        repo_root = Path(__file__).resolve().parent.parent
        spec_path = repo_root / args.spec

    exit_code = run_gates(
        spec_path=spec_path,
        target_gate=args.gate,
        list_only=args.list,
        fast=args.fast,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
