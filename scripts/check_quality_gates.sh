#!/usr/bin/env bash
# ==============================================================================
# pyDoppelgangerHunt Quality Gates Runner Wrapper (POSIX Shell)
# ==============================================================================
# Delegates execution to scripts/run_quality_gates.py using gates.toml as the
# single source of truth.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [ -z "${PYTHON:-}" ]; then
    if command -v python >/dev/null 2>&1; then
        PYTHON="python"
    elif command -v python.exe >/dev/null 2>&1; then
        PYTHON="python.exe"
    elif command -v py >/dev/null 2>&1; then
        PYTHON="py"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON="python3"
    else
        PYTHON="python"
    fi
fi

cd "${REPO_ROOT}"
exec "$PYTHON" "scripts/run_quality_gates.py" "$@"
