# 16DOF-WheelLeg-RL

[![CI](https://github.com/wusi321/16DOF-WheelLeg-RL/actions/workflows/main.yml/badge.svg)](https://github.com/wusi321/16DOF-WheelLeg-RL/actions/workflows/main.yml)

16 自由度轮足机器人（Unitree **go2w** 模型）的强化学习训练工程。

训练算法移植自 `rc_mjlab` 的 16DOF 轮足任务（PPO + 非对称 Actor-Critic，
mjlab / MuJoCo-Warp / RSL-RL），机器人模型替换为 `go2w/go2w.xml`。
地形配置沿用源工程的 Flat / Rough / Crawl 三套任务；比赛地图
`Robocon2027.xml` 暂未接入，等确定用法后再接。

## 演示

梅花桩任务（`Robot-StepStone-v0`）的回放视频（详细曲线分析见 [`docs/TRAINING.md`](docs/TRAINING.md)）。

**本轮（`2026-10-27_07-56-35`，600 迭代）**——同一份权重在曲线"飘掉"前后的对照：

| 阶段 | 表现 | 视频 |
| --- | --- | --- |
| 500 迭代（切换前） | 正常过桩，覆盖率在爬升 | 🎬 [stepstone_demo_500.mp4](assets/videos/2026-10-27_07-56-35/stepstone_demo_500.mp4)（0.5 MB） |
| 600 迭代（切换后） | 覆盖率 0.85，但**末期趴在桩上磨时间**、不再过终点 | 🎬 [stepstone_demo_600.mp4](assets/videos/2026-10-27_07-56-35/stepstone_demo_600.mp4)（6.2 MB） |

**上一轮（`2026-10-26_22-26-00`，500 迭代）**——旧奖励下的"高速飞越"：

| 阶段 | 表现 | 视频 |
| --- | --- | --- |
| 全程 | 课程全通，但**靠滞空飞过**（覆盖率 0.43） | 🎬 [stepstone_demo.mp4](assets/videos/2026-10-26_22-26-00/stepstone_demo.mp4)（2.8 MB） |

> 其余任务的回放可以用 `uv run play <Task> --num-envs 12` 自己录；
> 梅花桩按难度看：`uv run python tools/play_stepstone_levels.py --checkpoint-file <ckpt>`。

## 训练记录

两轮长训练的曲线解读、行为分析和对应的奖励改动，单独记在
**[docs/TRAINING.md](docs/TRAINING.md)**；截图在 `assets/images/<运行时间>/`，视频在
`assets/videos/<运行时间>/`。
## 目录结构

```text
16DOF-WheelLeg-RL/
├─ go2w/                     # Unitree go2w MJCF 原始模型（未修改）
│  ├─ go2w.xml               #   16 关节：4 腿 x (hip, thigh, calf, wheel)
│  ├─ scene.xml              #   官方平地场景
│  ├─ scene_terrain.xml      #   官方高程图场景
│  └─ assets/                #   网格
├─ Robocon2027.xml           # 比赛地图（stepping_stones 段已接入训练，见 docs/STEPSTONE.md）
├─ assets/                   # 演示素材（按运行时间归档）
│  ├─ videos/<运行时间>/      #   回放视频（500 迭代 / 600 迭代各一段）
│  └─ images/<运行时间>/      #   训练曲线截图
├─ src/robot/                # ★ 移植的强化学习算法
│  ├─ __init__.py            #   任务注册（Flat / Rough / Crawl / StepStone）
│  ├─ naming.py              # ★ go2w 名称/位姿/几何常量（唯一来源）
│  ├─ robot_cfg.py           # ★ go2w 装配：执行器、初始位姿、控制增益
│  ├─ config/env_cfgs.py     #   观测/动作/奖励/终止/域随机化/地形
│  ├─ config/rl_cfg.py       #   PPO 超参（原样）
│  ├─ mdp/                   #   自定义奖励、课程、指令、低通动作、扰动
│  │  └─ stepstone.py        #   梅花桩专用：进度、轮滑、成功/跌落、课程
│  ├─ terrains/              #   RC 高墙 / 低杆 / 梅花桩地形
│  │  └─ stepping_stones.py  #   交错桩列（桩高/中心距/错位/缺桩随难度）
│  └─ rsl_rl/                #   HIM 代码（本工程中不可用，见下文）
├─ mjlab/                    # 固定版本的 mjlab 框架（本地可编辑依赖，含 2 处补丁）
├─ tools/
│  ├─ check_go2w_contract.py # 离线校验：名称模式 / 关节限位 / 站立高度（只要 mujoco）
│  ├─ check_stepstone_coverage.py # 单环境验证“每根桩都要踩到”的判定与奖励
│  ├─ plot_stepstone_terrain.py   # 不训练、不出图：按难度画梅花桩俯视图（查间距）
│  ├─ play_stepstone_levels.py    # 按难度回放：每个环境钉到 level 0..N-1
│  ├─ smoke_build_env.py     # 四个任务 build+reset+step（CPU 可跑）
│  └─ smoke_train_ppo.py     # 1 次 PPO 迭代（CPU 可跑，验证训练链路）
├─ docs/
│  ├─ PORTING.md             # 移植说明：映射表、实测数值、与源工程的差异
│  ├─ STEPSTONE.md           # 梅花桩任务：地形/课程/奖励/终止的完整说明
│  └─ TRAINING.md            # 两轮长训练的曲线分析、行为退化与奖励修复记录
├─ setup_env.sh              # 一键测速换源 + 环境检查（Linux / WSL / Git Bash）
├─ pyproject.toml
└─ uv.lock                   # 源工程的依赖锁定
```

## 环境要求

- Python 3.10 – 3.13，[uv](https://docs.astral.sh/uv/)
- **NVIDIA GPU**（mjlab 通过 MuJoCo-Warp 在 GPU 上并行 2048 个环境）
- 见 `DEPENDENCIES.md`：MuJoCo 3.8、`mjlab[cu128]`、PyTorch CUDA 12.8
- Linux 无显示器环境：mjlab 默认 `MUJOCO_GL=egl`，而 `import mujoco` 会立刻加载
  EGL 后端。缺库时报
  `AttributeError: 'NoneType' object has no attribute 'eglQueryString'`，装一下系统库：

  ```bash
  sudo apt-get install -y --no-install-recommends libegl1 libgl1 libosmesa6
  # 或者改用软件渲染：export MUJOCO_GL=osmesa
  ```

## 安装与训练

在工程根目录执行（Windows 用 Git Bash，Linux/WSL 用系统 bash）：

```bash
bash setup_env.sh --install   # ① 测速选最快 PyPI 源 + ② 环境检查 + ③ uv sync
bash setup_env.sh             # 只做 ①+②，不动依赖
bash setup_env.sh --check     # 只做环境检查
```

`setup_env.sh` 会并行测速 7 个常用镜像（清华 / 阿里 / 中科大 / 腾讯 / 华为 /
南大 / 官方），把最快的写进 `pyproject.toml` 的默认 `[[tool.uv.index]]`，
不会动 `pytorch-cu128`、`nvidia` 两个显式源，也不会让 `uv.lock` 失效。
**如果机器上没有 uv，脚本会自动安装**（Windows 走官方 PowerShell 脚本 →
winget；Linux/WSL 走官方 `install.sh`；兜底 `python -m pip install --user uv`），
装好之后继续后面的换源/检查/安装流程。
其它选项：`--deep`（再跑模型契约 + 场景冒烟）、`--pip`（顺带写 pip 源）、
`--restore`（恢复清华源）、`--dry-run`（只测速，不写文件也不装 uv）、
`--list`、`--help`。

装好之后：

```bash
uv run train Robot-Flat-v0
uv run train Robot-Rough-v0
uv run train Robot-Crawl-v0
uv run train Robot-StepStone-v0
uv run play  Robot-Rough-v0
uv run play  Robot-StepStone-v0 --checkpoint-file logs/rsl_rl/robot_stepstone/<run>/model_650.pt
uv run python tools/play_stepstone_levels.py --checkpoint-file <ckpt>   # 按难度看（每个环境一个等级）
```

| Task ID | 用途 | episode |
| --- | --- | --- |
| `Robot-Flat-v0` | 平地基础运动 | 20 s |
| `Robot-Rough-v0` | 台阶、随机网格、高墙、坡面、Perlin | 20 s |
| `Robot-Crawl-v0` | 低杆 / 低姿态匍匐 | 30 s |
| `Robot-StepStone-v0` | 梅花桩（`Robocon2027.xml` 的 stepping_stones 段） | 30 s |

### 梅花桩任务（Robot-StepStone-v0）

按 `16DOF_RL.md` 的规划实现，完整说明见 [docs/STEPSTONE.md](docs/STEPSTONE.md)：

- **地形**：单桩交错（zigzag，0.20 × 0.20 m 方桩，左右交替 ±0.20 m），patch
  **8.0 × 1.6 m**（通行方向 2026-10-28 由 4.0 m 翻倍），路程 6.8 m；
  桩高 0.06 → 0.20 m、每跳中心距 0.43 → 0.46 m（`max_step` 封顶）、位置噪声
  0 → ±0.03 m；桩区默认是 0.6 m 深坑，掉下去即失败。逐级俯视图见
  [docs/images/terrain_levels.png](docs/images/terrain_levels.png)。
- **课程**：10 档难度对应讨论里的 5 个阶段，通过整段地形升级、走不到 25% 降级；
  速度指令随跟踪率自适应上调（0.3–0.5 → 最高 0.8 m/s）。
- **奖励**：实际前进进度（不用轮速）+ 速度/偏航跟踪 + 姿态/高度约束 + 轮滑 +
  力矩/功率/动作变化率 + 机身碰撞惩罚 + 到达终点 `+120`（按桩覆盖率打折）、
  另按赛事规则加了逐桩约束——新踩上一根桩 `+2`、踩满整段 `+20`、≥2 个轮子触坑底
  每步 `-2`（`src/robot/mdp/stone_coverage.py`）。
- **观测**：actor 326 维 = 53 维本体状态 + 273 维局部高度扫描（21 × 13，1.6 × 1.0 m）。
- **终止**：超时、倾角 > 60°、机身触地、机身过低、到达终点（成功）、NaN。

## 控制与观测契约

| 项目 | 值 |
| --- | --- |
| 物理步长 | `0.005 s`（200 Hz），`decimation=4` → 策略 50 Hz |
| 并行环境 | 2048 起，实训练 20480（8 m 地形下 40960 会 OOM，见下） |
| 腿执行器 | 位置控制，`Kp=50.0`、`Kd=1.5` |
| 轮执行器 | 速度控制，`Kd=1.0` |
| 力矩上限 | hip/thigh `23.7 Nm`、calf `45.43 Nm`、wheel `15 Nm`（取自 go2w.xml） |
| 默认站姿 | hip `0`、thigh `0.9`、calf `-1.8`、机身高度 `0.36 m` |
| 动作缩放 | 外展 `0.125 rad`、其余腿关节 `0.25 rad`、轮速 `5.0 rad/s` |
| 动作维度 | 16：[12 个腿关节位置] + [4 个轮速] |
| Actor 观测 | Flat/Rough/Crawl：53（角速度 3 + 投影重力 3 + 指令 3 + 腿位置 12 + 腿速度 12 + 轮速 4 + 上一步动作 16）<br>StepStone：**326** = 53 + 高度扫描 273（21 × 13，1.6 × 1.0 m，0.08 m 分辨率） |
| Critic 观测 | 额外：机身线速度 3 + 轮地接触 4 + 高度扫描 273（StepStone 的 actor 也已包含高度扫描） |
| 关节顺序 | `FL, FR, RL, RR` × `(hip, thigh, calf, wheel)` |
| 接触配额 | `nconmax=128`、`njmax=256`；`ccd_iterations=16`。**不能低于 8**：实测 `<8` 时 warp 完全不生成轮-地接触（0 条），机器人会穿地；`≥8` 与 80 的接触完全一致。也不能设太大：warp 的 EPA 缓冲按 `ccd_iterations` 线性放大，40960 环境 × `ccd=80` 要 22.8 GB（会 OOM），`16` 只要 5.3 GB |
| 显存经验值 | 48 GiB 卡上，**8 m 梅花桩地形（391 个 geom）× 40960 环境会 OOM**（`wp_cuda_graph_launch` 报 Warp CUDA error 2）。降到 **20480 环境**可稳定训练（约 101 次/24 h）；想保留 40960 就用 `--env.scene.terrain.terrain-generator.num-rows 6`（geom 391 → 255） | 

## 校验移植是否与模型一致

不需要 GPU：

```bash
# 1) 名称/位姿/几何与 go2w 模型的一致性（只需要 mujoco）
uv run --no-project --with mujoco python tools/check_go2w_contract.py

# 2) 四个任务能否 build / reset / step（CPU 后端，首次会编译 mujoco-warp 内核，较慢）
uv run python tools/smoke_build_env.py

# 3) 完整训练链路：1 次 PPO 迭代，日志写到 logs/smoke_ppo
uv run python tools/smoke_train_ppo.py
```

第 1 条检查：MJCF 电机是否已移除、名称正则是否命中 12 腿关节 / 4 轮 / 4 轮体、
站立与匍匐位姿是否都在关节限位内、正运动学下轮子最低点是否正好落地、
是否有机器人 geom 还留在 group 0（会让高度扫描打到自身）。

日志默认写本地 TensorBoard（`logs/rsl_rl/...`）。mjlab 上游默认是 wandb，
需要 `wandb login`；如果要切回去：

```bash
uv run train Robot-Rough-v0 --agent.logger wandb
```

## CI（GitHub Actions）

`.github/workflows/main.yml` 三个 job：

| Job | 触发 | 内容 |
| --- | --- | --- |
| 静态检查 | push / PR | `compileall`、工作流 YAML 解析、`shellcheck` |
| go2w 模型契约 | push / PR | `uv lock --check` + 名称/限位/落地高度校验（只要 mujoco） |
| 训练链路冒烟（CPU） | push 到 main / 手动 | `uv sync` + 四任务 build/step + 一次 PPO 迭代 |

托管 runner 没有 GPU，冒烟用的是 mujoco-warp 的 CPU 后端，只验证“装得上、跑得通”，
不做真正的长时间训练。

## 说明

- `go2w/` 与 `Robocon2027.xml` 未做任何修改。
- 原名 `mjcf/wheelleg.xml` 的机器人、`model_rough.pt`、`model_6800.onnx`
  属于旧机器人，未移植也不兼容（观测/动作顺序不同）。
- `src/robot/rsl_rl/` 中的 HIM 训练代码依赖外部 `him_mjlab` 包，源工程归档里
  没有该包，**当前无法运行**；实际训练路径是 mjlab 自带的 PPO。
  详见 `docs/PORTING.md`。

## 感知接口（Odin1）

真机感知（留形科技 Odin1：点云/IMU/里程计）与仿真高度扫描的统一接口、sim2real 差异建模、
局部地形观测定义和三阶段落地计划，见 [`docs/perception/`](docs/perception/README.md)。

实现位于根目录 `perception/`（**纯 numpy，不依赖 ROS 2，也不依赖 mjlab**），训练环境不受影响：

| 模块 | 作用 |
| --- | --- |
| `perception/frame.py` | `TerrainFrame` 契约 + 栅格常量（13×21 @ 0.08 m，base 系 yaw 对齐） |
| `perception/encode.py` | 冻结的 273 维观测编码（`clip(-1,1)`，未命中 → 1.0） |
| `perception/sources/sim.py` | MuJoCo 射线传感器 → `TerrainFrame` |
| `perception/sources/replay.py` | `.npz` 帧包读写与回放（第二阶段离线验证用） |
| `perception/augment.py` | 噪声/丢点/FOV 掩膜/平滑/延迟（训练侧 sim2real 增强） |

```bash
# 单元测试（42 项，CPU，约 0.3 s）
uv run python -m unittest discover -s tests -t .

# 黄金测试：编码器与训练观测逐元素比对（需要 CPU 后端，首次编译较慢）
uv run python tools/check_perception_alignment.py
```

2026-10-26 实测：Flat / StepStone 各 12 步 × 2 环境，`max|diff| = 0.0`。
