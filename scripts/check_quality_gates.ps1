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

[CmdletBinding()]
param(
    [switch]$Fast
)

$ErrorActionPreference = "Stop"

function Write-GateHeader {
    param([string]$GateNum, [string]$Title)
    Write-Host "`n========================================================" -ForegroundColor Cyan
    Write-Host " [GATE $GateNum] $Title" -ForegroundColor Cyan
    Write-Host "========================================================" -ForegroundColor Cyan
}

function Write-GateSuccess {
    param([string]$GateNum, [string]$Title)
    Write-Host "[PASSED] Gate $($GateNum): $Title`n" -ForegroundColor Green
}

function Write-GateFailure {
    param([string]$GateNum, [string]$Title, [string]$ErrorMsg)
    Write-Host "`n[FAILED] Gate $($GateNum): $Title" -ForegroundColor Red
    Write-Host "Error details: $ErrorMsg`n" -ForegroundColor Red
    exit 1
}

$startTime = Get-Date

# --- GATE 1: MYPY TYPE CHECKING ---
Write-GateHeader "1" "Static Type Safety (Mypy: pydoppelgangerhunt, tests)"
try {
    python -m mypy pydoppelgangerhunt tests
    if ($LASTEXITCODE -ne 0) { throw "Mypy reported type errors." }
    Write-GateSuccess "1" "Static Type Safety (Mypy)"
} catch {
    Write-GateFailure "1" "Static Type Safety (Mypy)" $_
}

# --- GATE 2: PYLINT CODE QUALITY (10.00 / 10.00) ---
Write-GateHeader "2" "Code Quality & Strict Linting (Pylint 10.00 / 10.00: pydoppelgangerhunt, tests)"
try {
    python -m pylint pydoppelgangerhunt tests
    if ($LASTEXITCODE -ne 0) { throw "Pylint score fell below 10.00 threshold." }
    Write-GateSuccess "2" "Code Quality & Strict Linting (Pylint)"
} catch {
    Write-GateFailure "2" "Code Quality & Strict Linting (Pylint)" $_
}

# --- GATE 3: DEPTRY DEPENDENCY HYGIENE ---
Write-GateHeader "3" "Dependency Packaging & Import Hygiene (Deptry)"
try {
    python -m deptry .
    if ($LASTEXITCODE -ne 0) { throw "Deptry detected dependency or import issues." }
    Write-GateSuccess "3" "Dependency Packaging & Import Hygiene (Deptry)"
} catch {
    Write-GateFailure "3" "Dependency Packaging & Import Hygiene (Deptry)" $_
}

# --- GATE 4: BANDIT SECURITY SCANNER ---
Write-GateHeader "4" "Static Application Security Testing (Bandit: pydoppelgangerhunt)"
try {
    python -m bandit -r pydoppelgangerhunt -ll -q
    if ($LASTEXITCODE -ne 0) { throw "Bandit reported security vulnerabilities." }
    Write-GateSuccess "4" "Static Application Security Testing (Bandit)"
} catch {
    Write-GateFailure "4" "Static Application Security Testing (Bandit)" $_
}

# --- GATE 5: PIP-AUDIT VULNERABILITY SCANNER ---
Write-GateHeader "5" "Dependency CVE Vulnerability Audit (pip-audit)"
try {
    python -m pip_audit . --local
    if ($LASTEXITCODE -ne 0) { throw "pip-audit detected known CVEs." }
    Write-GateSuccess "5" "Dependency CVE Vulnerability Audit (pip-audit)"
} catch {
    Write-GateFailure "5" "Dependency CVE Vulnerability Audit (pip-audit)" $_
}

# --- GATE 6: PYTEST TEST SUITE & COVERAGE ---
Write-GateHeader "6" "Test Suite & Code Coverage Verification (Pytest >= 85%)"
try {
    python -m pytest -q --cov=pydoppelgangerhunt --cov-report=term-missing --cov-report=xml
    if ($LASTEXITCODE -ne 0) { throw "Pytest reported test failures or coverage fell below threshold." }
    Write-GateSuccess "6" "Test Suite & Code Coverage Verification (Pytest)"
} catch {
    Write-GateFailure "6" "Test Suite & Code Coverage Verification (Pytest)" $_
}

# --- GATE 7: DOGFOODING CODE CLONE BARRIER (DUAL-TIER) ---
Write-GateHeader "7" "Dogfooding Code Clone Barrier (Dual-Tier Macro + Micro)"
try {
    Write-Host "Running Tier 1: Macro Structural Barrier (--threshold 0.70 --min-lines 8 --sliding-window --idioms)..." -ForegroundColor Yellow
    python -m pydoppelgangerhunt.cli pydoppelgangerhunt --threshold 0.70 --min-lines 8 --sliding-window --idioms
    if ($LASTEXITCODE -ne 0) { throw "pyDoppelgangerHunt identified macro structural redundancy (>= 70%)." }

    Write-Host "Running Tier 2: Micro Sub-Block Barrier (--threshold 0.80 --min-lines 6 --idioms)..." -ForegroundColor Yellow
    python -m pydoppelgangerhunt.cli pydoppelgangerhunt --threshold 0.80 --min-lines 6 --idioms
    if ($LASTEXITCODE -ne 0) { throw "pyDoppelgangerHunt identified micro sub-block redundancy (>= 80%)." }

    Write-GateSuccess "7" "Dogfooding Code Clone Barrier (Dual-Tier Macro + Micro)"
} catch {
    Write-GateFailure "7" "Dogfooding Code Clone Barrier" $_
}

$elapsed = (Get-Date) - $startTime
Write-Host "========================================================" -ForegroundColor Green
Write-Host (" ALL QUALITY GATES PASSED CLEANLY in {0:N1}s!" -f $elapsed.TotalSeconds) -ForegroundColor Green
Write-Host "========================================================`n" -ForegroundColor Green
