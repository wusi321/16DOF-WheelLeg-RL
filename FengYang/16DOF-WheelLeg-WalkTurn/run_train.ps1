# Train a 16DOF wheel-legged task.
#
# Usage:
#   .\run_train.ps1                                              # default turn task, 2048 envs
#   .\run_train.ps1 Robot-WalkTurn-Stand-v0 --agent.max-iterations 300
#   .\run_train.ps1 Robot-WalkTurn-Straight-v0 --env.scene.num-envs 2048 --agent.max-iterations 1200 --agent.run-name straight_v1
#   .\run_train.ps1 Robot-WalkTurn-v0 --agent.resume True --agent.load-run <run-dir> --agent.load-checkpoint model_600.pt
# Note: with --agent.resume, --agent.max-iterations is the number of ADDITIONAL iterations.
#
# mjlab parses nested configs with tyro, so overrides are --env.xxx / --agent.xxx.

param(
    [string]$Task = "Robot-WalkTurn-v0",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Args
)

$ErrorActionPreference = "Continue"
$ProjectRoot = $PSScriptRoot
$Python = "D:\Data\RC\RC_QiangHuaXueXi\16DOF-WheelLeg-RL-main\.venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "Python not found: $Python (venv from 16DOF-WheelLeg-RL-main)"
}

$env:PYTHONPATH = Join-Path $ProjectRoot "src"
$env:PYTHONIOENCODING = "utf-8"
Set-Location $ProjectRoot

Write-Host "==> training task $Task" -ForegroundColor Cyan
Write-Host "    project : $ProjectRoot"
Write-Host "    python  : $Python"

& $Python scripts\train.py $Task @Args
exit $LASTEXITCODE