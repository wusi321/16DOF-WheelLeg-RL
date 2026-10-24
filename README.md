# 16DOF-WheelLeg-RL

16 自由度轮足机器人（Unitree **go2w** 模型）的强化学习训练工程。

训练算法移植自 `rc_mjlab` 的 16DOF 轮足任务（PPO + 非对称 Actor-Critic，
mjlab / MuJoCo-Warp / RSL-RL），机器人模型替换为 `go2w/go2w.xml`。
地形配置沿用源工程的 Flat / Rough / Crawl 三套任务；比赛地图
`Robocon2027.xml` 暂未接入，等确定用法后再接。

## 目录结构

```text
16DOF-WheelLeg-RL/
├─ go2w/                     # Unitree go2w MJCF 原始模型（未修改）
│  ├─ go2w.xml               #   16 关节：4 腿 x (hip, thigh, calf, wheel)
│  ├─ scene.xml              #   官方平地场景
│  ├─ scene_terrain.xml      #   官方高程图场景
│  └─ assets/                #   网格
├─ Robocon2027.xml           # 比赛地图（待接入）
├─ src/robot/                # ★ 移植的强化学习算法
│  ├─ __init__.py            #   任务注册（Robot-Flat/Rough/Crawl-v0）
│  ├─ naming.py              # ★ go2w 名称/位姿/几何常量（唯一来源）
│  ├─ robot_cfg.py           # ★ go2w 装配：执行器、初始位姿、控制增益
│  ├─ config/env_cfgs.py     #   观测/动作/奖励/终止/域随机化/地形
│  ├─ config/rl_cfg.py       #   PPO 超参（原样）
│  ├─ mdp/                   #   自定义奖励、课程、指令、低通动作、扰动
│  ├─ terrains/              #   RC 高墙 / 低杆等比赛地形
│  └─ rsl_rl/                #   HIM 代码（本工程中不可用，见下文）
├─ mjlab/                    # 固定版本的 mjlab 框架（本地可编辑依赖，含 2 处补丁）
├─ tools/
│  ├─ check_go2w_contract.py # 离线校验：名称模式 / 关节限位 / 站立高度（只要 mujoco）
│  ├─ smoke_build_env.py     # 三个任务 build+reset+step（CPU 可跑）
│  └─ smoke_train_ppo.py     # 1 次 PPO 迭代（CPU 可跑，验证训练链路）
├─ docs/PORTING.md           # 移植说明：映射表、实测数值、与源工程的差异
├─ pyproject.toml
└─ uv.lock                   # 源工程的依赖锁定
```

## 环境要求

- Python 3.10 – 3.13，[uv](https://docs.astral.sh/uv/)
- **NVIDIA GPU**（mjlab 通过 MuJoCo-Warp 在 GPU 上并行 2048 个环境）
- 见 `DEPENDENCIES.md`：MuJoCo 3.8、`mjlab[cu128]`、PyTorch CUDA 12.8

## 安装与训练

在工程根目录执行：

```bash
uv sync                    # 安装依赖（会自动安装本地 mjlab）
uv run train Robot-Flat-v0
uv run train Robot-Rough-v0
uv run train Robot-Crawl-v0
uv run play  Robot-Rough-v0
```

| Task ID | 用途 | episode |
| --- | --- | --- |
| `Robot-Flat-v0` | 平地基础运动 | 20 s |
| `Robot-Rough-v0` | 台阶、随机网格、高墙、坡面、Perlin | 20 s |
| `Robot-Crawl-v0` | 低杆 / 低姿态匍匐 | 30 s |

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
| Actor 观测 | 53：角速度 3 + 投影重力 3 + 指令 3 + 腿位置 12 + 腿速度 12 + 轮速 4 + 上一步动作 16 |
| Critic 观测 | 额外：机身线速度 3 + 轮地接触 4 + 高度扫描 |
| 关节顺序 | `FL, FR, RL, RR` × `(hip, thigh, calf, wheel)` |

## 校验移植是否与模型一致

不需要 GPU：

```bash
# 1) 名称/位姿/几何与 go2w 模型的一致性（只需要 mujoco）
uv run --no-project --with mujoco python tools/check_go2w_contract.py

# 2) 三个任务能否 build / reset / step（CPU 后端，首次会编译 mujoco-warp 内核，较慢）
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

## 说明

- `go2w/` 与 `Robocon2027.xml` 未做任何修改。
- 原名 `mjcf/wheelleg.xml` 的机器人、`model_rough.pt`、`model_6800.onnx`
  属于旧机器人，未移植也不兼容（观测/动作顺序不同）。
- `src/robot/rsl_rl/` 中的 HIM 训练代码依赖外部 `him_mjlab` 包，源工程归档里
  没有该包，**当前无法运行**；实际训练路径是 mjlab 自带的 PPO。
  详见 `docs/PORTING.md`。
