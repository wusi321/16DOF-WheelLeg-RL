$ErrorActionPreference='Stop'
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
& 'C:\Users\dell\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' (Join-Path $PSScriptRoot 'run_sim.py')
