#!/usr/bin/env bash
# ==============================================================================
# pyDoppelgangerHunt Quality Gates & Invariant Barrier Validation Script
# ==============================================================================
# Enforces strict repository standards across:
#   1. Static Type Checking (Mypy: pydoppelgangerhunt, tests)
#   2. Code Quality & Strict Linting (Pylint 10.00 / 10.00)
#   3. Dependency & Packaging Hygiene (Deptry)
#   4. Static Application Security Testing (Bandit)
#   5. Supply Chain CVE Vulnerability Audit (pip-audit)
#   6. Unit & Feature Test Suite with Coverage (Pytest >= 85%)
#   7. Dogfooding Code Clone Barrier (Dual-Tier Macro + Micro)
# ==============================================================================

set -euo pipefail

# ANSI color codes
CYAN='\033[0;36m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

write_gate_header() {
    local gate_num="$1"
    local title="$2"
    echo -e "\n${CYAN}========================================================${NC}"
    echo -e "${CYAN} [GATE ${gate_num}] ${title}${NC}"
    echo -e "${CYAN}========================================================${NC}"
}

write_gate_success() {
    local gate_num="$1"
    local title="$2"
    echo -e "${GREEN}[PASSED] Gate ${gate_num}: ${title}${NC}\n"
}

write_gate_failure() {
    local gate_num="$1"
    local title="$2"
    local error_msg="${3:-unknown}"
    echo -e "\n${RED}[FAILED] Gate ${gate_num}: ${title}${NC}"
    echo -e "${RED}Error details: ${error_msg}${NC}\n"
    exit 1
}

START_TIME=$(date +%s)

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

# --- GATE 1: MYPY TYPE CHECKING ---
write_gate_header "1" "Static Type Safety (Mypy: pydoppelgangerhunt, tests)"
if "$PYTHON" -m mypy pydoppelgangerhunt tests; then
    write_gate_success "1" "Static Type Safety (Mypy)"
else
    write_gate_failure "1" "Static Type Safety (Mypy)" "Mypy reported type errors."
fi

# --- GATE 2: PYLINT CODE QUALITY (10.00 / 10.00) ---
write_gate_header "2" "Code Quality & Strict Linting (Pylint 10.00 / 10.00: pydoppelgangerhunt, tests)"
if "$PYTHON" -m pylint pydoppelgangerhunt tests; then
    write_gate_success "2" "Code Quality & Strict Linting (Pylint)"
else
    write_gate_failure "2" "Code Quality & Strict Linting (Pylint)" "Pylint score fell below 10.00 threshold."
fi

# --- GATE 3: DEPTRY DEPENDENCY HYGIENE ---
write_gate_header "3" "Dependency Packaging & Import Hygiene (Deptry)"
if "$PYTHON" -m deptry .; then
    write_gate_success "3" "Dependency Packaging & Import Hygiene (Deptry)"
else
    write_gate_failure "3" "Dependency Packaging & Import Hygiene (Deptry)" "Deptry detected dependency or import issues."
fi

# --- GATE 4: BANDIT SECURITY SCANNER ---
write_gate_header "4" "Static Application Security Testing (Bandit: pydoppelgangerhunt)"
if "$PYTHON" -m bandit -r pydoppelgangerhunt -ll -q; then
    write_gate_success "4" "Static Application Security Testing (Bandit)"
else
    write_gate_failure "4" "Static Application Security Testing (Bandit)" "Bandit reported security vulnerabilities."
fi

# --- GATE 5: PIP-AUDIT VULNERABILITY SCANNER ---
write_gate_header "5" "Dependency CVE Vulnerability Audit (pip-audit)"
if "$PYTHON" -m pip_audit . --local; then
    write_gate_success "5" "Dependency CVE Vulnerability Audit (pip-audit)"
else
    write_gate_failure "5" "Dependency CVE Vulnerability Audit (pip-audit)" "pip-audit detected known CVEs."
fi

# --- GATE 6: PYTEST TEST SUITE & COVERAGE ---
write_gate_header "6" "Test Suite & Code Coverage Verification (Pytest >= 85%)"
if "$PYTHON" -m pytest -q --cov=pydoppelgangerhunt --cov-report=term-missing --cov-report=xml; then
    write_gate_success "6" "Test Suite & Code Coverage Verification (Pytest)"
else
    write_gate_failure "6" "Test Suite & Code Coverage Verification (Pytest)" "Pytest reported test failures or coverage fell below threshold."
fi

# --- GATE 7: DOGFOODING CODE CLONE BARRIER (DUAL-TIER) ---
write_gate_header "7" "Dogfooding Code Clone Barrier (Dual-Tier Macro + Micro)"
echo -e "${YELLOW}Running Tier 1: Macro Structural Barrier (--threshold 0.70 --min-lines 8 --sliding-window --idioms)...${NC}"
if ! "$PYTHON" -m pydoppelgangerhunt.cli pydoppelgangerhunt --threshold 0.70 --min-lines 8 --sliding-window --idioms; then
    write_gate_failure "7" "Dogfooding Code Clone Barrier" "pyDoppelgangerHunt identified macro structural redundancy (>= 70%)."
fi

echo -e "${YELLOW}Running Tier 2: Micro Sub-Block Barrier (--threshold 0.80 --min-lines 6 --idioms)...${NC}"
if ! "$PYTHON" -m pydoppelgangerhunt.cli pydoppelgangerhunt --threshold 0.80 --min-lines 6 --idioms; then
    write_gate_failure "7" "Dogfooding Code Clone Barrier" "pyDoppelgangerHunt identified micro sub-block redundancy (>= 80%)."
fi
write_gate_success "7" "Dogfooding Code Clone Barrier (Dual-Tier Macro + Micro)"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo -e "${GREEN}========================================================${NC}"
echo -e "${GREEN} ALL QUALITY GATES PASSED CLEANLY in ${ELAPSED}s!${NC}"
echo -e "${GREEN}========================================================\n${NC}"
