# 16DOF-WheelLeg-RL

[![CI](https://github.com/wusi321/16DOF-WheelLeg-RL/actions/workflows/main.yml/badge.svg)](https://github.com/wusi321/16DOF-WheelLeg-RL/actions/workflows/main.yml)

16 自由度轮足机器人（Unitree **go2w** 模型）的强化学习训练工程。

训练算法移植自 `rc_mjlab` 的 16DOF 轮足任务（PPO + 非对称 Actor-Critic，
mjlab / MuJoCo-Warp / RSL-RL），机器人模型替换为 `go2w/go2w.xml`。
地形配置沿用源工程的 Flat / Rough / Crawl 三套任务；比赛地图
`Robocon2027.xml` 暂未接入，等确定用法后再接。

## 演示

梅花桩任务（`Robot-StepStone-v0`）的回放视频：

- 🎬 **[assets/videos/2026-10-26_22-26-00/stepstone_demo.mp4](assets/videos/2026-10-26_22-26-00/stepstone_demo.mp4)**（原始文件名 `创飞梅花桩.mp4`，2.8 MB）

> 其余任务的回放可以用 `uv run play <Task> --num-envs 12` 自己录；
> 梅花桩按难度看：`uv run python tools/play_stepstone_levels.py --checkpoint-file <ckpt>`。

## 训练曲线

以 `2026-10-26_22-26-00` 这次 500 迭代的 StepStone 训练为例（wandb 导出，完整截图在
[`assets/images/2026-10-26_22-26-00/`](assets/images/2026-10-26_22-26-00/)）。

| 曲线 | 读法 |
| --- | --- |
| <img src="assets/images/2026-10-26_22-26-00/1.png" width="720"><br>**地形课程** | `level_cap` 每约 50 迭代线性放行一级；`mean/max` 最终到 9；`downgraded` 在 50 迭代后归零，说明不再需要降级 |
| <img src="assets/images/2026-10-26_22-26-00/2.png" width="720"><br>**速度指令课程** | `distance` 稳定在 2.8（=终点距离）；`x_tracking_ema` 一路涨到 3.0 —— **这是异常值**，说明它测的是"实际速度"而不是"跟踪率"（已在 `bc47895` 修正）；`x_range_max` 因此被打满到 0.8 |
| <img src="assets/images/2026-10-26_22-26-00/3.png" width="720"><br>**任务指标** | `stone_coverage` 只有 **0.43** —— 近一半的桩没踩到；`wheel_slip` 涨到 1.3（空转严重）；`mean_leg_action_acc` 从 1.8 降到 1.1（动作变平滑） |
| <img src="assets/images/2026-10-26_22-26-00/4.png" width="720"><br>**奖励项（一）** | `track_lin_vel_x` 在 100 迭代后掉到 ~0 —— 不再跟速度指令；`stone_touch` 稳定在 0.07 左右（被"飞越"稀释） |
| <img src="assets/images/2026-10-26_22-26-00/5.png" width="720"><br>**奖励项（二）** | `forward_progress` 爬到 0.25 后走平；`goal_reached` 稳定在 0.05（频繁成功）；`is_terminated` 稳定在 −0.01 |
| <img src="assets/images/2026-10-26_22-26-00/6.png" width="720"><br>**终止原因** | `goal_reached` 终止数一路涨到 ~650/迭代；`time_out` 在 50 迭代后归零；每次升级难度会看到 `base_too_low` / `bad_orientation` 的尖峰（摔下去），随后收敛 |
| <img src="assets/images/2026-10-26_22-26-00/7.png" width="720"><br>**训练损失** | `Loss/value` 平稳（升级时尖峰是分布变化）；`entropy` 从 22.5 单调降到 11（探索收缩）；`learning_rate` 自适应下降 |
| <img src="assets/images/2026-10-26_22-26-00/8.png" width="720"><br>**总览** | `mean_reward` 爬到 8.5 后基本走平；`mean_episode_length` 从 300 步压到 **58 步** —— 回合越来越短，正是"高速冲过去"的直接证据 |

> 这组曲线对应的是**修复前**的策略（`c64bb45` 之前的奖励）：课程能一路升到 9 级、成功率很高，
> 但 `stone_coverage` 只有 0.43、回合长度 58 步。想让策略真正"踩满每根桩"，需要
> `c64bb45` 之后的奖励（终点按覆盖率打折 + 滞空/超速惩罚）重新训练。

## 目录结构

```text
16DOF-WheelLeg-RL/
├─ go2w/                     # Unitree go2w MJCF 原始模型（未修改）
│  ├─ go2w.xml               #   16 关节：4 腿 x (hip, thigh, calf, wheel)
│  ├─ scene.xml              #   官方平地场景
│  ├─ scene_terrain.xml      #   官方高程图场景
│  └─ assets/                #   网格
├─ Robocon2027.xml           # 比赛地图（stepping_stones 段已接入训练，见 docs/STEPSTONE.md）
├─ assets/                   # 演示素材
│  ├─ videos/                #   回放视频（stepstone_demo.mp4）
│  └─ images/                #   训练曲线截图（自行添加）
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
│  ├─ smoke_build_env.py     # 四个任务 build+reset+step（CPU 可跑）
│  └─ smoke_train_ppo.py     # 1 次 PPO 迭代（CPU 可跑，验证训练链路）
├─ docs/
│  ├─ PORTING.md             # 移植说明：映射表、实测数值、与源工程的差异
│  └─ STEPSTONE.md           # 梅花桩任务：地形/课程/奖励/终止的完整说明
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
```

| Task ID | 用途 | episode |
| --- | --- | --- |
| `Robot-Flat-v0` | 平地基础运动 | 20 s |
| `Robot-Rough-v0` | 台阶、随机网格、高墙、坡面、Perlin | 20 s |
| `Robot-Crawl-v0` | 低杆 / 低姿态匍匐 | 30 s |
| `Robot-StepStone-v0` | 梅花桩（`Robocon2027.xml` 的 stepping_stones 段） | 20 s |

### 梅花桩任务（Robot-StepStone-v0）

按 `16DOF_RL.md` 的规划实现，完整说明见 [docs/STEPSTONE.md](docs/STEPSTONE.md)：

- **地形**：两条交错桩列（0.20 × 0.20 m 方桩、横向中心距 0.40 m ≈ 轮距），
  桩高 0.06 → 0.20 m、前后中心距 0.30 → 0.60 m、错位 0 → 0.30 m、位置噪声
  0 → ±0.03 m、缺桩 0 → 10%，最终与实场一致；桩区默认是 0.6 m 深坑，掉下去即失败。
- **课程**：10 档难度对应讨论里的 5 个阶段，通过整段地形升级、走不到 25% 降级；
  速度指令随跟踪率自适应上调（0.3–0.5 → 最高 0.8 m/s）。
- **奖励**：实际前进进度（不用轮速）+ 速度/偏航跟踪 + 姿态/高度约束 + 轮滑 +
  力矩/功率/动作变化率 + 机身碰撞惩罚 + 到达终点 `+60`，摔落/撞地 `-50`；
  另按赛事规则加了逐桩约束——新踩上一根桩 `+3`、踩满整段 `+10`、≥2 个轮子触坑底
  每步 `-2`（`src/robot/mdp/stone_coverage.py`）。
- **观测**：actor 326 维 = 53 维本体状态 + 273 维局部高度扫描（21 × 13，1.6 × 1.0 m）。
- **终止**：超时、倾角 > 60°、机身触地、机身过低、到达终点（成功）、NaN。

## 控制与观测契约

| 项目 | 值 |
| --- | --- |
| 物理步长 | `0.005 s`（200 Hz），`decimation=4` → 策略 50 Hz |
| 并行环境 | 2048 |
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
