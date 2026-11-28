function Initialize-MHEnvironment {
    param([Parameter(Mandatory=$true)][string]$PackageRoot)
    if ([IO.Path]::GetPathRoot($PackageRoot) -ieq 'C:\') {
        throw '请先把 MH 放到 D/F 等非 C 盘目录，再安装或运行。'
    }
    $mhCache = $env:MH_CACHE_ROOT
    if (-not $mhCache) {
        $mhCache = Join-Path $PackageRoot '.cache'
        if ($mhCache -match '[^\x00-\x7F]') {
            # NVRTC/Warp 1.12 cannot compile into a non-ASCII path on Windows.
            # Keep runtime caches inside THIS Git checkout, with an ASCII path.
            $mhRepo = Get-Item -LiteralPath $PackageRoot
            while ($mhRepo -and -not (Test-Path -LiteralPath (Join-Path $mhRepo.FullName '.git'))) {
                $mhRepo = $mhRepo.Parent
            }
            if (-not $mhRepo -or $mhRepo.FullName -match '[^\x00-\x7F]') {
                throw 'CUDA缓存需要英文路径：请把整个MH复制到非C盘英文目录，或设置MH_CACHE_ROOT为项目内的英文缓存路径。'
            }
            $mhCache = Join-Path $mhRepo.FullName '.cache\MH'
        }
    }
    $mhCache = [IO.Path]::GetFullPath($mhCache)
    if ([IO.Path]::GetPathRoot($mhCache) -ieq 'C:\' -or $mhCache -match '[^\x00-\x7F]') {
        throw 'MH_CACHE_ROOT必须是非C盘、全英文的绝对路径。'
    }
    foreach ($name in @('TMP','TEMP','TMPDIR')) {
        Set-Item "Env:$name" (Join-Path $mhCache 'tmp')
    }
    foreach ($item in @{
        WARP_CACHE_PATH='warp'; CUDA_CACHE_PATH='cuda'; MPLCONFIGDIR='mpl';
        TORCHINDUCTOR_CACHE_DIR='torchinductor'; TRITON_CACHE_DIR='triton';
        TORCH_HOME='torch'; TORCH_EXTENSIONS_DIR='torch_extensions';
        UV_CACHE_DIR='uv'; PIP_CACHE_DIR='pip'; XDG_CACHE_HOME='xdg'
    }.GetEnumerator()) {
        Set-Item "Env:$($item.Key)" (Join-Path $mhCache $item.Value)
    }
    New-Item -ItemType Directory -Force -Path $env:TMP | Out-Null
    $env:PYTHONPATH = (Join-Path $PackageRoot 'vendor\mjlab\src') + [IO.Path]::PathSeparator + (Join-Path $PackageRoot 'vendor\mxz\src')
    $env:PYTHONIOENCODING = 'utf-8'
    $env:PYTHONUNBUFFERED = '1'
    $env:PYTHONUTF8 = '1'
    $env:PYTHONDONTWRITEBYTECODE = '1'
    $env:PYTHONNOUSERSITE = '1'
    $env:PIP_DISABLE_PIP_VERSION_CHECK = '1'
}
