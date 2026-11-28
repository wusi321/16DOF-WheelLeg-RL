# Evaluate a trained policy on fixed velocity commands.
#
# Usage:
#   .\run_eval.ps1                                                # newest turn checkpoint, all scenarios
#   .\run_eval.ps1 Robot-WalkTurn-Straight-v0                     # newest straight checkpoint
#   .\run_eval.ps1 Robot-WalkTurn-Stand-v0 --scenario stand
#   .\run_eval.ps1 Robot-WalkTurn-v0 -Checkpoint logs\...\model_1200.pt --scenario turn_left,turn_right
#   .\run_eval.ps1 Robot-WalkTurn-v0 --video
#
# Extra arguments are forwarded to tools/eval_policy.py.

param(
    [string]$Task = "Robot-WalkTurn-v0",
    [string]$Checkpoint = "",
    [string[]]$Scenario = @("all"),
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

if ($Checkpoint -eq "") {
    $logRoot = switch ($Task) {
        "Robot-WalkTurn-Stand-v0"    { Join-Path $ProjectRoot "logs\rsl_rl\robot_walk_turn_stand" }
        "Robot-WalkTurn-Straight-v0" { Join-Path $ProjectRoot "logs\rsl_rl\robot_walk_turn_straight" }
        default                      { Join-Path $ProjectRoot "logs\rsl_rl\robot_walk_turn" }
    }
    $latest = Get-ChildItem -Path $logRoot -Recurse -Filter "model_*.pt" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($null -eq $latest) {
        throw "No checkpoint under $logRoot. Run run_train.ps1 first."
    }
    $Checkpoint = $latest.FullName
}

Write-Host "==> evaluating $Task" -ForegroundColor Cyan
Write-Host "    checkpoint : $Checkpoint"
$ScenarioText = $Scenario -join ","
Write-Host "    scenario   : $ScenarioText"

& $Python tools\eval_policy.py --task $Task --checkpoint $Checkpoint --scenario $ScenarioText @Args
exit $LASTEXITCODE