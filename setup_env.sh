#!/usr/bin/env bash
# =============================================================================
# setup_env.sh — 一键「测速换源 + 环境检查」
#
# 用法:
#   bash setup_env.sh              # 测速选源 → 写入项目 → 环境检查
#   bash setup_env.sh --mirror     # 只测速换源
#   bash setup_env.sh --check      # 只做环境检查
#   bash setup_env.sh --install    # 换源 + 检查 + uv sync
#   bash setup_env.sh --deep       # 额外跑 go2w 契约校验 + 场景冒烟
#   bash setup_env.sh --pip        # 同时写 pip 源
#   bash setup_env.sh --list       # 只列出候选镜像
#   bash setup_env.sh --restore    # 恢复默认源（清华）
#   bash setup_env.sh --dry-run    # 只测速，不写任何文件
#   bash setup_env.sh --yes        # 不询问，全部自动确认
#
# 支持 Linux / WSL / Windows Git Bash。写入的文件:
#   pyproject.toml            第一个 [[tool.uv.index]]（默认源）的 url
#   %APPDATA%\pip\pip.ini     仅 --pip，Windows
#   ~/.config/pip/pip.conf    仅 --pip，Linux/WSL
#
# 说明: 只改「默认源」的 URL，不会动 pytorch-cu128 / nvidia 这两个显式源，
#       也不会让 uv.lock 失效（uv 仍按 pypi.org 记录 canonical 源）。
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
PYPROJECT="$PROJECT_ROOT/pyproject.toml"

# ---------------------------------------------------------------- 颜色与输出
if [ -t 1 ] && [ "${NO_COLOR:-}" = "" ]; then
  C_OK=$'\033[32m'; C_WARN=$'\033[33m'; C_FAIL=$'\033[31m'
  C_DIM=$'\033[2m'; C_RST=$'\033[0m'
else
  C_OK=""; C_WARN=""; C_FAIL=""; C_DIM=""; C_RST=""
fi

OK_N=0; WARN_N=0; FAIL_N=0
PY_USABLE=1   # .venv 是否可在当前平台使用（由 check_python_env 设置）

section() { printf '\n%s\n' "${C_DIM}── $* ──${C_RST}"; }
ok()      { OK_N=$((OK_N + 1));   printf '  %s %s\n' "${C_OK}[ OK ]${C_RST}" "$*"; }
warn()    { WARN_N=$((WARN_N + 1)); printf '  %s %s\n' "${C_WARN}[WARN]${C_RST}" "$*"; }
bad()     { FAIL_N=$((FAIL_N + 1)); printf '  %s %s\n' "${C_FAIL}[FAIL]${C_RST}" "$*"; }
note()    { printf '  %s %s\n' "${C_DIM}[ -- ]${C_RST}" "$*"; }
die()     { printf '%s\n' "${C_FAIL}$*${C_RST}" >&2; exit 1; }

have() { command -v "$1" >/dev/null 2>&1; }

# ---------------------------------------------------------------- 参数解析
MODE="all"          # all | mirror | check | list | restore
DO_INSTALL=0
DO_DEEP=0
DO_PIP=0
DRY_RUN=0
ASSUME_YES=0

usage() {
  cat <<'EOF'
用法: bash setup_env.sh [选项]

  (无参数)      测速选最快镜像 → 写入项目 → 环境检查
  --mirror      只测速换源
  --check       只做环境检查
  --install     换源 + 检查 + uv sync
  --deep        额外跑 go2w 契约校验 + 场景冒烟
  --pip         同时写 pip 源
  --list        列出候选镜像与当前源
  --restore     恢复默认源（清华 TUNA）
  --dry-run     只测速，不写任何文件
  --yes, -y     不询问，全部自动确认
  -h, --help    显示本帮助
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --mirror)  MODE="mirror" ;;
    --check)   MODE="check" ;;
    --list)    MODE="list" ;;
    --restore) MODE="restore" ;;
    --install) DO_INSTALL=1 ;;
    --deep)    DO_DEEP=1 ;;
    --pip)     DO_PIP=1 ;;
    --dry-run) DRY_RUN=1 ;;
    --yes|-y)  ASSUME_YES=1 ;;
    -h|--help) usage; exit 0 ;;
    *) die "未知参数: $1（用 --help 查看用法）" ;;
  esac
  shift
done

[ -f "$PYPROJECT" ] || die "找不到 $PYPROJECT —— 请在项目根目录运行本脚本。"

# ---------------------------------------------------------------- 平台识别
UNAME_S="$(uname -s 2>/dev/null || echo unknown)"
case "$UNAME_S" in
  MINGW*|MSYS*|CYGWIN*) PLATFORM="windows-gitbash"; IS_MSYS=1 ;;
  *)                    IS_MSYS=0 ;;
esac
if [ "$IS_MSYS" = 0 ]; then
  if [ -r /proc/version ] && grep -qi microsoft /proc/version 2>/dev/null; then
    PLATFORM="wsl"
  elif [ "$UNAME_S" = "Linux" ]; then
    PLATFORM="linux"
  elif [ "$UNAME_S" = "Darwin" ]; then
    PLATFORM="macos"
  else
    PLATFORM="unknown"
  fi
fi

# ---------------------------------------------------------------- 镜像表
# 只测 PyPI 镜像（本项目其余两个源是 pytorch/nvidia 专用，不可替换）
MIRROR_NAMES=(tuna aliyun ustc tencent huawei nju pypi)
MIRROR_URLS=(
  "https://pypi.tuna.tsinghua.edu.cn/simple"
  "https://mirrors.aliyun.com/pypi/simple"
  "https://pypi.mirrors.ustc.edu.cn/simple"
  "https://mirrors.cloud.tencent.com/pypi/simple"
  "https://repo.huaweicloud.com/repository/pypi/simple"
  "https://mirror.nju.edu.cn/pypi/web/simple"
  "https://pypi.org/simple"
)
MIRROR_LABELS=(
  "清华 TUNA"
  "阿里云"
  "中科大 USTC"
  "腾讯云"
  "华为云"
  "南京大学"
  "官方 PyPI"
)
DEFAULT_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"

PROBE_MAX_TIME="${PROBE_MAX_TIME:-6}"   # 单镜像最长探测时间(秒)

current_index_url() {
  grep -m1 -E '^url[[:space:]]*=' "$PYPROJECT" 2>/dev/null | sed -E 's/.*"(.*)".*/\1/'
}

current_index_name() {
  awk '/^\[\[tool\.uv\.index\]\]/ { b = 1; next }
       b && /^name[[:space:]]*=/ { gsub(/.*"|".*/, ""); print; exit }' "$PYPROJECT" 2>/dev/null
}

# ---------------------------------------------------------------- 测速
# 并行探测每个镜像的 /simple/numpy/（约 1.3 MB，比索引页更能反映真实吞吐），
# 失败的项再串行重试一次。结果格式:
#   score<TAB>name<TAB>url<TAB>label<TAB>http<TAB>bytes<TAB>reason
probe_mirrors() {
  local work
  work="$(mktemp -d 2>/dev/null || mktemp -d -t setupenv)" || return 1
  local i
  for i in "${!MIRROR_URLS[@]}"; do
    (
      local url="${MIRROR_URLS[$i]%/}/numpy/"
      local res code size t score attempt reason=""
      score=0; code=000; size=0; t=0
      for attempt in 1 2; do
        if have curl; then
          res="$(curl -sSL --connect-timeout 4 --max-time "$PROBE_MAX_TIME" \
                   -o /dev/null -w '%{http_code} %{size_download} %{time_total}' \
                   "$url" 2>/dev/null)" || res="000 0 0"
        else
          res="000 0 0"
        fi
        read -r code size t <<<"$res"
        code="${code:-000}"; size="${size:-0}"; t="${t:-0}"
        [ "$code" = "200" ] && [ "$size" -gt 10000 ] && break
        [ "$attempt" = 1 ] && sleep 0.3
      done
      if [ "$code" = "200" ] && [ "$size" -gt 10000 ]; then
        score="$(awk -v s="$size" -v tt="$t" 'BEGIN{ if (tt+0 < 0.001) tt=0.001; printf "%d", (s+0)/tt }')"
      else
        score=0
        if [ "$code" = "000" ]; then reason="连接失败"; else reason="HTTP $code"; fi
      fi
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$score" "${MIRROR_NAMES[$i]}" "${MIRROR_URLS[$i]}" "${MIRROR_LABELS[$i]}" \
        "$code" "$size" "$reason" > "$work/p$i"
    ) &
  done
  wait
  cat "$work"/p* 2>/dev/null | sort -t "$(printf '\t')" -k1,1nr
  rm -rf "$work"
}

human_rate() { # bytes/sec -> 人类可读
  awk -v b="${1:-0}" 'BEGIN{
    if (b >= 1048576)      printf "%.2f MB/s", b/1048576
    else if (b >= 1024)    printf "%.1f KB/s", b/1024
    else                   printf "%d B/s", b
  }'
}

list_mirrors() {
  printf '%-3s %-14s %s\n' "#" "名称" "URL"
  local i
  for i in "${!MIRROR_NAMES[@]}"; do
    printf '%-3s %-14s %s\n' "$((i + 1))" "${MIRROR_LABELS[$i]}" "${MIRROR_URLS[$i]}"
  done
  printf '\n当前项目默认源: %s\n' "$(current_index_url)"
}

# ---------------------------------------------------------------- 写源
# apply_index_url <url> <显示名> <index 名>
apply_index_url() {
  local new_url="$1" label="$2" idx_name="$3"
  local backup="$PYPROJECT.bak"
  cp "$PYPROJECT" "$backup" || return 1
  awk -v target="$new_url" -v idx="$idx_name" '
    BEGIN { inblock = 0; done = 0 }
    /^\[\[tool\.uv\.index\]\]/ { inblock = 1; print; next }
    inblock && !done && /^name[[:space:]]*=/ {
      print "name = \"" idx "\""; next
    }
    inblock && !done && /^url[[:space:]]*=/ {
      print "url = \"" target "\""; done = 1; inblock = 0; next
    }
    { print }
  ' "$backup" > "$PYPROJECT" || { mv "$backup" "$PYPROJECT"; return 1; }
  printf '  已写入 %s（备份: %s）\n' "$label" "$(basename "$backup")"
}

restore_index_url() {
  if [ "$(current_index_url)" = "$DEFAULT_INDEX_URL" ] && [ "$(current_index_name)" = "tsinghua" ]; then
    ok "默认源已经是 $DEFAULT_INDEX_URL"
  else
    apply_index_url "$DEFAULT_INDEX_URL" "清华 TUNA" "tsinghua"
    ok "已恢复默认源: $DEFAULT_INDEX_URL"
  fi
  rm -f "$PYPROJECT.bak"
  return 0
}

write_pip_conf() {
  local url="$1" host conf
  host="$(printf '%s' "$url" | sed -E 's#^https?://##; s#/.*$##')"
  if [ "$IS_MSYS" = 1 ]; then
    conf="${APPDATA:-$HOME}/pip/pip.ini"
  else
    conf="${XDG_CONFIG_HOME:-$HOME/.config}/pip/pip.conf"
  fi
  mkdir -p "$(dirname "$conf")" || { warn "无法创建目录 $(dirname "$conf")"; return 1; }
  [ -f "$conf" ] && cp "$conf" "$conf.bak"
  {
    printf '[global]\n'
    printf 'index-url = %s\n' "$url"
    printf 'trusted-host = %s\n' "$host"
  } > "$conf"
  ok "pip 源已写入 $conf"
}

# ---------------------------------------------------------------- 换源主流程
run_mirror() {
  section "镜像测速（每项最多 ${PROBE_MAX_TIME}s）"
  if ! have curl; then
    bad "缺少 curl，无法测速（Linux: apt install curl；Git Bash 自带 curl）"
    return 1
  fi

  local results=""
  results="$(probe_mirrors)"

  printf '  %-14s %-12s %-6s %s\n' "镜像" "下载速率" "HTTP" "URL"
  printf '  %s\n' "-------------------------------------------------------------------"
  local best_score="" best_name="" best_url="" best_label=""
  local min_score=51200   # 低于 50 KB/s 视为“太慢”，不推荐
  while IFS=$'\t' read -r score name url label code size reason; do
    [ -z "${score:-}" ] && continue
    local mark=""
    if [ "$score" -gt 0 ] && [ "$score" -ge "$min_score" ]; then
      if [ -z "$best_score" ] || [ "$score" -gt "$best_score" ]; then
        best_score="$score"; best_name="$name"; best_url="$url"; best_label="$label"
      fi
    elif [ "$score" -gt 0 ]; then
      mark=" ${C_DIM}(偏慢)${C_RST}"
    else
      mark=" ${C_DIM}(${reason:-不可用})${C_RST}"
    fi
    printf '  %-14s %-12s %-6s %s%s\n' "$label" "$(human_rate "$score")" "$code" "$url" "$mark"
  done <<<"$results"

  if [ -z "$best_score" ]; then
    bad "所有镜像都不可用或过慢 —— 检查网络 / DNS / 代理；可用 --list 查看候选源"
    return 1
  fi

  printf '\n  最快: %s（%s）\n' "$best_label" "$(human_rate "$best_score")"

  local current
  current="$(current_index_url)"
  if [ "$current" = "$best_url" ]; then
    ok "当前默认源就是最快的，无需修改"
    return 0
  fi

  if [ "$DRY_RUN" = 1 ]; then
    note "--dry-run：不写入。建议源 $best_url"
    return 0
  fi

  if [ "$ASSUME_YES" = 0 ] && [ -t 0 ]; then
    printf '  切换 uv 默认源 %s → %s ? [Y/n] ' "$(printf '%.48s' "$current")" "$best_url"
    local ans=""; read -r ans || ans=""
    case "${ans:-Y}" in
      [nN]*) note "已跳过换源"; return 0 ;;
    esac
  fi

  apply_index_url "$best_url" "$best_label" "$best_name"

  # 默认源 URL 变化通常不破坏 lock；若镜像版本差异导致需要重锁，自动补上。
  if have uv; then
    if uv lock --check >/dev/null 2>&1; then
      ok "uv.lock 仍然有效，无需重锁"
    else
      note "uv.lock 需要按新源重新解析，执行 uv lock ..."
      if (cd "$PROJECT_ROOT" && uv lock >/dev/null 2>&1); then
        ok "uv.lock 已更新"
      else
        warn "uv lock 失败，稍后可手动执行: uv lock"
      fi
    fi
  fi

  if [ "$DO_PIP" = 1 ]; then
    write_pip_conf "$best_url"
  else
    note "未改动 pip 配置（需要时加 --pip）"
  fi
  return 0
}

# ---------------------------------------------------------------- 环境检查
check_file() {
  local path="$1" label="$2"
  if [ -e "$PROJECT_ROOT/$path" ]; then
    ok "$label"
  else
    bad "缺少 $path"
  fi
}

check_basics() {
  section "基础工具"
  ok "shell: bash ${BASH_VERSION} / $(uname -srm)"
  case "$PLATFORM" in
    windows-gitbash) note "平台: Windows Git Bash" ;;
    wsl)             note "平台: WSL" ;;
    linux)           note "平台: Linux" ;;
    macos)           note "平台: macOS（训练需另行确认 mujoco-warp 支持）" ;;
    *)               warn "未识别的平台: $UNAME_S" ;;
  esac

  have git  && ok "git: $(git --version 2>/dev/null)" || bad "缺少 git"
  have curl && ok "curl: $(curl --version 2>/dev/null | head -1 | awk '{print $2}')" || warn "缺少 curl（测速/网络检查会跳过）"
  have awk  && ok "awk: $(awk --version 2>/dev/null | head -1 || echo 可用)" || bad "缺少 awk"
  have uv   && ok "uv: $(uv --version 2>/dev/null)" || bad "缺少 uv —— 安装: curl -LsSf https://astral.sh/uv/install.sh | sh"

  if [ "$PLATFORM" = "windows-gitbash" ]; then
    if printf '%s' "$PROJECT_ROOT" | LC_ALL=C grep -q '[^ -~]'; then
      warn "项目路径含非 ASCII 字符（$PROJECT_ROOT）：Windows 上 MuJoCo 无法直接打开此类路径，本项目已用内存加载规避，但建议尽量放到纯英文路径"
    else
      ok "项目路径为纯 ASCII"
    fi
  fi
}

check_hardware() {
  section "硬件与磁盘"
  local cpus=""
  if have nproc; then cpus="$(nproc)"; elif [ "$IS_MSYS" = 1 ]; then cpus="${NUMBER_OF_PROCESSORS:-?}"; else cpus="?"; fi
  ok "CPU 核心: $cpus"

  local mem_gb=""
  if [ -r /proc/meminfo ]; then
    mem_gb="$(awk '/MemTotal/{printf "%.1f", $2/1048576}' /proc/meminfo)"
  fi
  [ -n "$mem_gb" ] && ok "内存: ${mem_gb} GB"

  local free_gb
  free_gb="$(df -Pk "$PROJECT_ROOT" 2>/dev/null | awk 'NR==2{printf "%.1f", $4/1048576}')"
  if [ -n "$free_gb" ]; then
    if awk -v g="$free_gb" 'BEGIN{exit !(g < 15)}'; then
      warn "磁盘可用 ${free_gb} GB，建议 ≥ 15 GB（torch cu128 约 2.6 GB + venv 约 5 GB + 日志）"
    else
      ok "磁盘可用: ${free_gb} GB"
    fi
  fi

  if have nvidia-smi; then
    local gpu
    gpu="$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | head -1)"
    if [ -n "$gpu" ]; then
      ok "GPU: $gpu"
      local cuda_drv
      cuda_drv="$(nvidia-smi 2>/dev/null | grep -o 'CUDA Version: [0-9.]*' | head -1 | awk '{print $3}')"
      [ -n "$cuda_drv" ] && ok "驱动支持的 CUDA: $cuda_drv（torch cu128 需要 ≥ 12.8 驱动）"
    else
      warn "nvidia-smi 无输出：驱动未装好或无 GPU"
    fi
  else
    warn "没有 nvidia-smi：只能做 CPU 冒烟（tools/smoke_build_env.py），正式训练需要 NVIDIA GPU"
  fi
}

check_network() {
  section "源与网络"
  local cur
  cur="$(current_index_url)"
  if [ -n "$cur" ]; then ok "uv 默认源: $cur"; else warn "无法从 pyproject.toml 解析默认源"; fi

  local p val
  for p in HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY http_proxy https_proxy all_proxy; do
    eval "val=\${$p:-}"
    [ -z "$val" ] && continue
    if [ "${#val}" -gt 60 ]; then val="$(printf '%s' "$val" | cut -c1-57)…"; fi
    note "代理 $p=$val"
  done
  [ -n "${PIP_INDEX_URL:-}" ] && note "PIP_INDEX_URL=$PIP_INDEX_URL"
  [ -n "${UV_DEFAULT_INDEX:-}" ] && warn "设置了 UV_DEFAULT_INDEX=$UV_DEFAULT_INDEX，会覆盖项目里的源"

  if have curl && [ -n "$cur" ]; then
    local code
    code="$(curl -sSL -o /dev/null --connect-timeout 3 --max-time 6 -w '%{http_code}' "${cur%/}/pip/" 2>/dev/null || echo 000)"
    if [ "$code" = "200" ]; then ok "当前默认源可访问（HTTP $code）"
    else warn "当前默认源访问异常（HTTP $code），可用 --mirror 换源"; fi
  fi

  # GitHub 只用于安装 mujoco-warp 的 git 依赖，用 smart-http 端点快速探测
  if have curl; then
    local gh_code
    gh_code="$(curl -sS -o /dev/null --connect-timeout 5 --max-time 8 \
      -w '%{http_code}' \
      'https://github.com/mujocolab/mjlab.git/info/refs?service=git-upload-pack' 2>/dev/null || echo 000)"
    case "$gh_code" in
      200|401) ok "GitHub 可访问（安装 mujoco-warp 的 git 依赖时需要）" ;;
      000)     warn "GitHub 连接超时：uv sync 拉取 mujoco-warp 可能失败；可 export HTTPS_PROXY=... 后重试" ;;
      *)       warn "GitHub 返回 HTTP $gh_code：uv sync 拉取 mujoco-warp 可能失败" ;;
    esac
  fi
}

check_project() {
  section "项目文件"
  check_file "pyproject.toml" "pyproject.toml"
  check_file "uv.lock" "uv.lock"
  check_file "go2w/go2w.xml" "go2w 模型"
  check_file "go2w/assets" "go2w 网格目录"
  check_file "src/robot/naming.py" "移植后的算法源码"
  check_file "mjlab/pyproject.toml" "本地 mjlab 框架"
  check_file "tools/check_go2w_contract.py" "自检脚本"
  if [ -e "$PROJECT_ROOT/Robocon2027.xml" ]; then
    ok "比赛地图 Robocon2027.xml（尚未接入训练）"
  else
    note "没有 Robocon2027.xml（地图接入前不影响训练）"
  fi

  if have uv; then
    if (cd "$PROJECT_ROOT" && uv lock --check >/dev/null 2>&1); then
      ok "uv.lock 与 pyproject.toml 一致"
    else
      warn "uv.lock 需要更新，执行: uv lock"
    fi
  fi
}

venv_kind() {
  if [ -x "$PROJECT_ROOT/.venv/Scripts/python.exe" ]; then echo windows
  elif [ -x "$PROJECT_ROOT/.venv/bin/python" ]; then echo posix
  else echo none; fi
}

check_python_env() {
  section "Python 环境"
  if ! have uv; then
    bad "没有 uv，跳过 Python 检查"
    PY_USABLE=0
    return
  fi

  local kind; kind="$(venv_kind)"
  case "$kind" in
    none)
      warn "还没有 .venv，运行: bash setup_env.sh --install（或 uv sync）"
      PY_USABLE=0
      return ;;
    windows)
      if [ "$IS_MSYS" = 1 ]; then
        ok ".venv 已存在（Windows 版）"
      else
        warn ".venv 是 Windows 版，不能在当前 Linux/WSL 里用；如需在 Linux 训练请设置 UV_PROJECT_ENVIRONMENT=.venv-linux 后 uv sync"
        PY_USABLE=0
        return
      fi ;;
    posix)
      if [ "$IS_MSYS" = 1 ]; then
        warn ".venv 是 Linux 版，不能在当前 Git Bash 里用；请用 --install 重建 Windows venv，或设置 UV_PROJECT_ENVIRONMENT"
        PY_USABLE=0
        return
      else
        ok ".venv 已存在（Linux 版）"
      fi ;;
  esac

  local info
  info="$(cd "$PROJECT_ROOT" && uv run --no-sync python - <<'PY' 2>/dev/null
import sys
print(f"python={sys.version.split()[0]}")
for mod in ("mujoco", "torch", "warp", "mjlab", "rsl_rl", "robot"):
    try:
        m = __import__(mod)
        print(f"{mod}={getattr(m, '__version__', 'ok')}")
    except Exception as exc:
        print(f"{mod}=MISSING({type(exc).__name__})")
try:
    import torch
    print(f"cuda_available={int(torch.cuda.is_available())}")
    print(f"torch_cuda={torch.version.cuda}")
except Exception:
    pass
PY
)" || true

  if [ -z "$(printf '%s' "$info" | grep -E '^python=')" ]; then
    bad "无法在 .venv 里运行 python（可能未安装完成），执行: uv sync"
    PY_USABLE=0
    return
  fi
  PY_USABLE=1

  local pyver="" cuda="" tcuda=""
  while IFS='=' read -r k v; do
    case "$k" in
      python)         pyver="$v" ;;
      cuda_available) cuda="$v" ;;
      torch_cuda)     tcuda="$v" ;;
      mujoco|torch|warp|mjlab|rsl_rl|robot)
        case "$v" in
          MISSING*) bad "缺少依赖 $k（执行 uv sync）" ;;
          *) ok "$k $v" ;;
        esac ;;
    esac
  done <<<"$info"

  [ -n "$pyver" ] && ok "venv Python $pyver（要求 3.10 ~ 3.13）"
  case "$pyver" in
    3.1[0-3].*) : ;;
    "") : ;;
    *) warn "Python $pyver 超出 pyproject 的 >=3.10,<3.14" ;;
  esac

  if [ "$cuda" = "1" ]; then
    ok "torch 检测到 CUDA（$tcuda）"
  else
    warn "torch 没检测到 CUDA（$tcuda）：只能 CPU 冒烟，正式训练需要 GPU 驱动"
  fi
}

check_tasks() {
  section "任务注册"
  if [ "$PY_USABLE" = 0 ]; then
    note "跳过任务注册检查（当前平台没有可用的 .venv）"
    return
  fi
  local tasks
  tasks="$(cd "$PROJECT_ROOT" && uv run --no-sync python - <<'PY' 2>/dev/null | tail -1
from mjlab.tasks.registry import list_tasks
print(" ".join(t for t in list_tasks() if t.startswith("Robot-")))
PY
)" || true
  if [ -z "$tasks" ]; then
    bad "没有注册到 Robot-* 任务（uv sync 是否完成？entry point 检查: src/robot.egg-info/entry_points.txt）"
    return
  fi
  local t
  for t in Robot-Flat-v0 Robot-Rough-v0 Robot-Crawl-v0; do
    case " $tasks " in
      *" $t "*) ok "已注册 $t" ;;
      *)        bad "未注册 $t" ;;
    esac
  done
}

check_deep() {
  section "深度检查（--deep）"
  if [ "$PY_USABLE" = 0 ] || [ "$(venv_kind)" = "none" ]; then
    warn "当前平台没有可用的 .venv，跳过深度检查"
    return
  fi

  local out
  out="$(cd "$PROJECT_ROOT" && uv run --no-sync python tools/check_go2w_contract.py 2>&1)" || true
  if printf '%s' "$out" | grep -q "all checks passed"; then
    ok "go2w 模型契约校验通过（名称/限位/落地高度）"
  else
    bad "go2w 模型契约校验失败，详见: uv run python tools/check_go2w_contract.py"
    printf '%s\n' "$out" | grep -E '^\[FAIL\]|Error|Traceback' | head -5 | sed 's/^/        /'
  fi

  note "场景冒烟需要编译 mujoco-warp 内核，首次约 2~4 分钟 ..."
  out="$(cd "$PROJECT_ROOT" && uv run --no-sync python tools/smoke_build_env.py 2>&1)" || true
  local n
  n="$(printf '%s' "$out" | grep -c "steps ok")"
  if [ "$n" -ge 3 ]; then
    ok "三个任务都能 build/reset/step（$n 个任务）"
  else
    bad "场景冒烟未全部通过（通过 $n/3），详见: uv run python tools/smoke_build_env.py"
    printf '%s\n' "$out" | grep -E 'Error|Traceback|Exception' | head -5 | sed 's/^/        /'
  fi
}

summary() {
  section "汇总"
  printf '  OK %d   WARN %d   FAIL %d\n' "$OK_N" "$WARN_N" "$FAIL_N"
  if [ "$FAIL_N" -gt 0 ]; then
    printf '\n%s\n' "${C_FAIL}存在必须修复的问题，见上面的 [FAIL]。${C_RST}"
    return 1
  fi
  if [ "$WARN_N" -gt 0 ]; then
    printf '\n%s\n' "${C_WARN}有告警但可以继续；训练前建议处理 [WARN]。${C_RST}"
  else
    printf '\n%s\n' "${C_OK}全部检查通过。${C_RST}"
  fi
  printf '\n下一步:\n'
  printf '  uv run train Robot-Flat-v0     # 训练（需要 NVIDIA GPU）\n'
  printf '  uv run play  Robot-Rough-v0    # 回放\n'
  printf '  bash setup_env.sh --deep       # 深度自检\n'
  return 0
}

# ---------------------------------------------------------------- 主流程
printf '%s\n' "项目: $PROJECT_ROOT"
printf '%s\n' "平台: $PLATFORM"

case "$MODE" in
  list)    list_mirrors; exit 0 ;;
  restore) restore_index_url; exit $? ;;
  mirror)  run_mirror || true ;;
  check)   : ;;
  all)     run_mirror || true ;;
esac

if [ "$MODE" != "mirror" ] || [ "$DO_INSTALL" = 1 ]; then
  check_basics
  check_hardware
  check_network
  check_project
  check_python_env
  check_tasks
  [ "$DO_DEEP" = 1 ] && check_deep
fi

if [ "$DO_INSTALL" = 1 ]; then
  section "安装依赖"
  if [ "$DRY_RUN" = 1 ]; then
    note "--dry-run：跳过 uv sync"
  else
    note "执行 uv sync（首次会下载 torch cu128，约 3 GB）..."
    if (cd "$PROJECT_ROOT" && uv sync); then
      ok "uv sync 完成"
    else
      bad "uv sync 失败（看上方日志；必要时 --mirror 换源 / 配置代理）"
    fi
  fi
fi

[ "$MODE" = "mirror" ] && [ "$DO_INSTALL" = 0 ] && exit 0

summary
