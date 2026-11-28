# Evaluate a trained policy on fixed velocity commands.
#
# Bundled checkpoint is used by default (checkpoints/model_1795.pt), so no
# training is required.
#
# Usage:
#   .\run_eval.ps1                                         # final locked turn policy, all scenarios
#   .\run_eval.ps1 --scenario stand
#   .\run_eval.ps1 Robot-WalkTurn-Stand-v0 --scenario stand
#   .\run_eval.ps1 --checkpoint checkpoints\model_1396.pt --scenario turn_left,turn_right
#   .\run_eval.ps1 --video
#
# Extra arguments are forwarded to tools/eval_policy.py. Everything is parsed
# by hand because PowerShell would otherwise bind "--foo" values positionally.

param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$AllArgs = @()
)

$ErrorActionPreference = "Continue"
$ProjectRoot = $PSScriptRoot
. "$ProjectRoot\tools\find_python.ps1"
$Python = Resolve-PythonExe -ProjectRoot $ProjectRoot

$Task = "Robot-WalkTurn-Locked-v0"
$Checkpoint = ""
$Scenario = "all"
$Extra = New-Object System.Collections.Generic.List[string]

$i = 0
$taskSeen = $false
while ($i -lt $AllArgs.Count) {
    $a = $AllArgs[$i]
    $hasNext = ($i + 1) -lt $AllArgs.Count
    if (($a -eq "-Task" -or $a -eq "--task") -and $hasNext) {
        $Task = $AllArgs[$i + 1]; $taskSeen = $true; $i += 2
    }
    elseif (($a -eq "-Checkpoint" -or $a -eq "--checkpoint") -and $hasNext) {
        $Checkpoint = $AllArgs[$i + 1]; $i += 2
    }
    elseif (($a -eq "-Scenario" -or $a -eq "--scenario") -and $hasNext) {
        $Scenario = $AllArgs[$i + 1]; $i += 2
    }
    elseif (-not $taskSeen -and $a -match "^Robot-") {
        $Task = $a; $taskSeen = $true; $i += 1
    }
    else {
        $Extra.Add($a); $i += 1
    }
}

$env:PYTHONPATH = Join-Path $ProjectRoot "src"
$env:PYTHONIOENCODING = "utf-8"
Set-Location $ProjectRoot

if ($Checkpoint -eq "") {
    $bundledMap = @{
        "Robot-WalkTurn-Stand-v0"    = "checkpoints\model_299.pt"
        "Robot-WalkTurn-Straight-v0" = "checkpoints\model_1396.pt"
        "Robot-WalkTurn-Right-v0"    = "checkpoints\model_1795.pt"
        "Robot-WalkTurn-v0"          = "checkpoints\model_1795.pt"
        "Robot-WalkTurn-Locked-v0"   = "checkpoints\model_1795.pt"
    }
    if ($bundledMap.ContainsKey($Task)) {
        $bundled = Join-Path $ProjectRoot $bundledMap[$Task]
    }
    else {
        $bundled = Join-Path $ProjectRoot "checkpoints\model_1795.pt"
    }

    if (Test-Path -LiteralPath $bundled) {
        $Checkpoint = $bundled
    }
    else {
        $logRoot = switch ($Task) {
            "Robot-WalkTurn-Stand-v0"    { Join-Path $ProjectRoot "logs\rsl_rl\robot_walk_turn_stand" }
            "Robot-WalkTurn-Straight-v0" { Join-Path $ProjectRoot "logs\rsl_rl\robot_walk_turn_straight" }
            "Robot-WalkTurn-Locked-v0"   { Join-Path $ProjectRoot "logs\rsl_rl\robot_walk_turn_locked" }
            default                      { Join-Path $ProjectRoot "logs\rsl_rl\robot_walk_turn" }
        }
        $latest = Get-ChildItem -Path $logRoot -Recurse -Filter "model_*.pt" -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 1
        if ($null -eq $latest) {
            throw "No checkpoint found. Missing $bundled and nothing under $logRoot. Train first (run_train.ps1) or pass --checkpoint <path>."
        }
        $Checkpoint = $latest.FullName
    }
}

Write-Host "==> evaluating $Task" -ForegroundColor Cyan
Write-Host "    checkpoint : $Checkpoint"
Write-Host "    scenario   : $Scenario"

& $Python tools\eval_policy.py --task $Task --checkpoint $Checkpoint --scenario $Scenario @Extra
exit $LASTEXITCODE