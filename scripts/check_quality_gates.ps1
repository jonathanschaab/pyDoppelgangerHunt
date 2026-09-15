# ==============================================================================
# pyDoppelgangerHunt Quality Gates Runner Wrapper (PowerShell)
# ==============================================================================
# Delegates execution to scripts/run_quality_gates.py using gates.toml as the
# single source of truth.
# ==============================================================================

[CmdletBinding()]
param(
    [string]$Gate,
    [switch]$Fast,
    [switch]$List
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$runnerPath = Join-Path $scriptDir "run_quality_gates.py"

$pyArgs = @()
if ($Gate) { $pyArgs += "--gate", $Gate }
if ($Fast) { $pyArgs += "--fast" }
if ($List) { $pyArgs += "--list" }

python $runnerPath @pyArgs
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
