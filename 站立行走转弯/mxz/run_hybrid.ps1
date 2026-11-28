# Hybrid controller demo in the native MuJoCo window -- the recommended quick look.
#
# Uses the bundled checkpoint checkpoints/model_1795.pt, so nothing has to be
# trained. Auto cycle: stand 2 s -> straight 0.4 -> turn left +0.8 ->
# straight 0.8 -> turn right -0.8 -> repeat.
#
# Usage:
#   .\run_hybrid.ps1                             # auto demo
#   .\run_hybrid.ps1 --manual                    # keyboard control from the start
#   .\run_hybrid.ps1 --distance 3.5              # robot clipped by the near plane
#   .\run_hybrid.ps1 --locked-checkpoint checkpoints\model_1795.pt
#   .\run_hybrid.ps1 --headless --steps 200      # no window, smoke test only
#
# Keys (viewer window must have focus): W/S vx, A/D wz, X stop, R auto.

param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Continue"
$ProjectRoot = $PSScriptRoot
. "$ProjectRoot\tools\find_python.ps1"
$Python = Resolve-PythonExe -ProjectRoot $ProjectRoot

$env:PYTHONPATH = Join-Path $ProjectRoot "src"
$env:PYTHONIOENCODING = "utf-8"
Set-Location $ProjectRoot

Write-Host "==> hybrid viewer (checkpoints/model_1795.pt)" -ForegroundColor Cyan
Write-Host "    python : $Python"

& $Python tools\view_hybrid.py @ExtraArgs
exit $LASTEXITCODE