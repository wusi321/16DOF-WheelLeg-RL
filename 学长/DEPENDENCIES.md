# 依赖说明

## Python 环境

- Python `>=3.10,<3.14`（本机验证用 3.12）
- `uv` 管理依赖
- MuJoCo `3.8.x`（`uv.lock` 锁 3.8.1 稳定版）
- `mjlab[cu128]` + PyTorch CUDA 12.8
- `pynput`

项目使用**本地可编辑**的 `mjlab`：

```toml
[tool.uv.sources]
mjlab = { path = "mjlab", editable = true }
```

## 与归档 `uv.lock` 的差异

源工程锁的是 MuJoCo `3.8.1.dev907589621`（nightly），该构建已从
`py.mujoco.org` 下架，`uv sync` 会报 404。本工程改为：

- `override-dependencies = ["mujoco>=3.8.0,<3.9"]`
- MuJoCo 从默认源取稳定版 `3.8.1`（PyPI），不再指向 `py.mujoco.org`
- `uv.lock` 已重新解析

其余依赖（torch cu128、warp-lang、mujoco-warp 的 git rev
`88b55fc2696960b927bc12584994bb8412b36558`、rsl-rl 5.4.0）保持不变。

## mjlab 来源与本地补丁

- 上游仓库：`https://github.com/mujocolab/mjlab.git`
- 基准提交：`40f8d93e31b589dccae78ba6aadfc4b74cd1e3fd`（2026-06-02）
- 版本：`1.4.0`，许可证 Apache-2.0，许可证保留在 `mjlab/LICENSE`
- 本地修改 2 处：
  1. `mjlab/src/mjlab/envs/mdp/dr/actuator.py` —— 为分组执行器补充
     名称到运行时执行器对象的解析，使 PD 增益和力矩限制随机化能作用于
     轮腿机器人的执行器组（源工程原样保留的补丁）。
  2. `mjlab/src/mjlab/scene/scene.py` —— `_load_scene_spec()`：
     当 `MjSpec.from_file()` 打不开 `scene.xml` 时改用文本解析。
     MuJoCo 在 Windows 上无法打开含非 ASCII 字符的路径，而本工程位于
     `D:\桌面\...`；不补这一处，任何场景都建不起来。

`mjlab/docs`、`.github`、`.claude`、`notebooks` 未复制，不影响运行。

## 硬件

- 训练：NVIDIA GPU（mjlab 通过 MuJoCo-Warp 在 GPU 上并行）。显存占用随
  `num_envs` 增长，默认 2048 个环境。
- 没有 NVIDIA GPU 时只能做静态校验，无法 `uv run train`。

## 基本入口

```bash
bash setup_env.sh --install    # 自动补装 uv → 测速换源 → 环境检查 → uv sync
uv sync
uv run train Robot-Flat-v0
uv run train Robot-Rough-v0
uv run train Robot-Crawl-v0
uv run play  Robot-Rough-v0

# 不依赖 GPU 的模型/名称一致性检查
uv run --no-project --with mujoco python tools/check_go2w_contract.py
```

> `setup_env.sh` 检测不到 `uv` 时会自动安装：Windows(Git Bash) 用官方
> `install.ps1`（失败再试 `winget install astral-sh.uv`）；Linux/WSL 用官方
> `install.sh`（curl 或 wget）；都不可用时回退 `python -m pip install --user uv`。
> 装完会重新扫描 `~/.local/bin` 等目录并继续执行，不需要重开终端。
