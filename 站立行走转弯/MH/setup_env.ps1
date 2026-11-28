param([string]$Python='python')
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\environment.ps1"
Initialize-MHEnvironment -PackageRoot $PSScriptRoot
Set-Location $PSScriptRoot
& $Python -c 'import sys; assert sys.version_info[:2] == (3,12), "Please use Python 3.12"'
if ($LASTEXITCODE -ne 0) { throw '需要 Python 3.12。可用 -Python 指定 python.exe 的完整路径。' }
& $Python -m venv "$PSScriptRoot\.venv"
if ($LASTEXITCODE -ne 0) { throw 'venv 创建失败' }
$mhPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
& $mhPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'pip 安装失败' }
& $mhPython -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch 安装失败' }
& $mhPython -m pip install -r "$PSScriptRoot\requirements.txt" -e "$PSScriptRoot\vendor\mjlab"
if ($LASTEXITCODE -ne 0) { throw '依赖安装失败' }
$env:MH_PYTHON = $mhPython
& "$PSScriptRoot\run.ps1" verify
Write-Host 'MH 环境已准备好。运行 .\run.ps1 打开原生 MuJoCo。'
