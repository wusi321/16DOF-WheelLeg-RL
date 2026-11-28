# Replay a trained policy in the viser web page (http://localhost:8080).
#
# Usage:
#   .\run_play.ps1                                                # newest checkpoint of turn task
#   .\run_play.ps1 Robot-WalkTurn-Straight-v0                     # newest checkpoint of straight task
#   .\run_play.ps1 Robot-WalkTurn-Stand-v0 --agent zero           # no policy, default standing pose
#   .\run_play.ps1 Robot-WalkTurn-v0 --checkpoint-file logs\...\model_1200.pt
#   .\run_play.ps1 Robot-WalkTurn-v0 --viewer native              # native MuJoCo window

param(
    [string]$Task = "Robot-WalkTurn-v0",
    [string]$Checkpoint = "",
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

if ($Checkpoint -ne "") {
    $Args = @("--checkpoint-file", $Checkpoint) + $Args
}
elseif (-not ($Args -match '^--(checkpoint-file|wandb-run-path|agent)')) {
    $logRoot = switch ($Task) {
        "Robot-WalkTurn-Stand-v0"    { Join-Path $ProjectRoot "logs\rsl_rl\robot_walk_turn_stand" }
        "Robot-WalkTurn-Straight-v0" { Join-Path $ProjectRoot "logs\rsl_rl\robot_walk_turn_straight" }
        default                      { Join-Path $ProjectRoot "logs\rsl_rl\robot_walk_turn" }
    }
    $latest = Get-ChildItem -Path $logRoot -Recurse -Filter "model_*.pt" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($null -eq $latest) {
        throw "No checkpoint under $logRoot. Run run_train.ps1 first, or pass --agent zero."
    }
    Write-Host "==> using newest checkpoint: $($latest.FullName)" -ForegroundColor Cyan
    $Args = @("--checkpoint-file", $latest.FullName) + $Args
}

& $Python scripts\play.py $Task @Args
exit $LASTEXITCODE