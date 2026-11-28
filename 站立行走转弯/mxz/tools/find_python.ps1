# Shared helper: locate a usable Python interpreter for the run_*.ps1 launchers.
#
# Resolution order:
#   1. $env:PYTHON                (explicit override, wins over everything)
#   2. <project>\.venv\Scripts\python.exe
#   3. <project>\venv\Scripts\python.exe
#   4. python on PATH
#
# Usage from a launcher in the project root:
#   . "$PSScriptRoot\tools\find_python.ps1"
#   $Python = Resolve-PythonExe -ProjectRoot $PSScriptRoot
#
# This file is ASCII-only on purpose: PowerShell 5.1 parses BOM-less UTF-8
# scripts with the ANSI code page and chokes on non-ASCII text.

function Resolve-PythonExe {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [string]$ProjectRoot
    )

    if ($env:PYTHON -and (Test-Path -LiteralPath $env:PYTHON)) {
        return (Resolve-Path -LiteralPath $env:PYTHON).Path
    }

    foreach ($candidate in @(
            (Join-Path $ProjectRoot ".venv\Scripts\python.exe"),
            (Join-Path $ProjectRoot "venv\Scripts\python.exe"))) {
        if (Test-Path -LiteralPath $candidate) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }

    $onPath = Get-Command python -ErrorAction SilentlyContinue
    if ($onPath -and $onPath.Source) {
        return $onPath.Source
    }

    throw (
        "Python interpreter not found. Pick one of these:`n" +
        "  1) create a venv in this folder:  python -m venv .venv   (then see README 1.1)`n" +
        "  2) point at an existing interpreter:  `$env:PYTHON = 'C:\path\to\python.exe'`n" +
        "  3) add python to PATH"
    )
}