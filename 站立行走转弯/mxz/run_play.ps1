# Replay a trained policy in a native MuJoCo window (default) or the viser web page.
#
# No training needed: checkpoints are bundled in checkpoints/
#   checkpoints/model_1795.pt  final locked-wheel turn policy (default)
#   checkpoints/model_299.pt   standing policy
#   checkpoints/model_1396.pt  older gait-style turn policy (reference)
#
# Usage:
#   .\run_play.ps1                                          # final locked turn policy, native window
#   .\run_play.ps1 Robot-WalkTurn-Stand-v0                   # standing policy
#   .\run_play.ps1 --viewer viser                            # viser web page (http://localhost:8080)
#   .\run_play.ps1 --agent zero                              # no policy, default standing pose
#   .\run_play.ps1 --checkpoint-file checkpoints\model_1795.pt
#   .\run_play.ps1 Robot-WalkTurn-Locked-v0 --checkpoint-file <path>
#
# Extra arguments are forwarded to scripts/play.py (tyro overrides). Arguments
# are parsed by hand so that "--viewer", "--agent" etc. are forwarded untouched.
# If you trained your own run, the newest checkpoint under logs/rsl_rl/ is used
# automatically when the bundled file is missing.
#
# Note: mjlab 1.4.0's viser joystick GUI raises AssertionError when a task's
# command range contains 0 (Stand / Straight / Locked tasks), so the native
# window is the default. --viewer viser works for Robot-WalkTurn-v0.

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

$hasOwnCheckpoint = $false
foreach ($e in $Extra) {
    if ($e -match "^--(checkpoint-file|wandb-run-path|agent)") { $hasOwnCheckpoint = $true }
}

$hasViewer = $false
foreach ($e in $Extra) {
    if ($e -match "^--viewer") { $hasViewer = $true }
}
if (-not $hasViewer) {
    # mjlab 1.4.0 viser GUI crashes on zero command ranges; native is safe.
    $Extra.Insert(0, "native")
    $Extra.Insert(0, "--viewer")
}

if ($Checkpoint -ne "") {
    $Extra.Insert(0, $Checkpoint)
    $Extra.Insert(0, "--checkpoint-file")
}
elseif (-not $hasOwnCheckpoint) {
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
        Write-Host "==> using bundled checkpoint: $bundled" -ForegroundColor Cyan
        $Extra.Insert(0, $bundled)
        $Extra.Insert(0, "--checkpoint-file")
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
            throw "No checkpoint found. Missing $bundled and nothing under $logRoot. Pass --agent zero to watch the default pose, or --checkpoint-file <path>."
        }
        Write-Host "==> using newest checkpoint: $($latest.FullName)" -ForegroundColor Cyan
        $Extra.Insert(0, $latest.FullName)
        $Extra.Insert(0, "--checkpoint-file")
    }
}

& $Python scripts\play.py $Task @Extra
exit $LASTEXITCODE