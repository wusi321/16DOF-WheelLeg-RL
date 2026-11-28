# Open a native MuJoCo window showing the calibrated 45-degree standing pose.
#
# No trained policy / checkpoint needed.
#   .\run_view.ps1                      # stand still (gravity-comp PD, nominal pose)
#   .\run_view.ps1 --wheel-speed 6       # wheels at 6 rad/s ~= 0.53 m/s, watch it roll
#   .\run_view.ps1 --thigh-offset 0.1    # try another pose: thigh targets +0.1 rad
#   .\run_view.ps1 --calf-offset -0.1    # calf targets -0.1 rad
#
# Mouse: left drag = rotate, right drag = pan, wheel = zoom. Close window to exit.

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

Write-Host "==> 45-degree stand viewer" -ForegroundColor Cyan
Write-Host "    python : $Python"

& $Python tools\view_stand.py @ExtraArgs
exit $LASTEXITCODE