param([Parameter(ValueFromRemainingArguments=$true)][string[]]$TaskArgs)
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\environment.ps1"
Initialize-MHEnvironment -PackageRoot $PSScriptRoot
$mhPython = $env:MH_PYTHON
if (-not $mhPython) { $mhPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $mhPython -PathType Leaf)) {
    throw '未找到 Python 环境。先运行 .\setup_env.ps1，或将 MH_PYTHON 指向已有兼容环境的 python.exe。'
}
Set-Location $PSScriptRoot
if (-not $TaskArgs) { $TaskArgs = @('view') }
if ($TaskArgs[0] -eq 'verify') {
    & $mhPython -B "$PSScriptRoot\tools\verify_package.py"
} else {
    & $mhPython -B "$PSScriptRoot\tools\mxz_robust.py" @TaskArgs
}
if ($LASTEXITCODE -ne 0) { throw "MH exited with code $LASTEXITCODE" }
