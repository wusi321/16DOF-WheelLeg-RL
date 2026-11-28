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
# Arguments are parsed by hand so that "--env.xxx" values are forwarded untouched.

param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$AllArgs = @()
)

$ErrorActionPreference = "Continue"
$ProjectRoot = $PSScriptRoot
. "$ProjectRoot\tools\find_python.ps1"
$Python = Resolve-PythonExe -ProjectRoot $ProjectRoot

$Task = "Robot-WalkTurn-v0"
$Extra = New-Object System.Collections.Generic.List[string]

if ($AllArgs.Count -gt 0 -and $AllArgs[0] -notlike "-*") {
    $Task = $AllArgs[0]
    for ($i = 1; $i -lt $AllArgs.Count; $i++) { $Extra.Add($AllArgs[$i]) }
}
else {
    for ($i = 0; $i -lt $AllArgs.Count; $i++) { $Extra.Add($AllArgs[$i]) }
}

$env:PYTHONPATH = Join-Path $ProjectRoot "src"
$env:PYTHONIOENCODING = "utf-8"
Set-Location $ProjectRoot

Write-Host "==> training task $Task" -ForegroundColor Cyan
Write-Host "    project : $ProjectRoot"
Write-Host "    python  : $Python"

& $Python scripts\train.py $Task @Extra
exit $LASTEXITCODE