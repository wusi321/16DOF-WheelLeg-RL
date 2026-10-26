# 移植说明（16DOF 轮足任务 → go2w 模型）

## 1. 来源与范围

- 算法来源：`D:\桌面\rc\RC_WheelLeg\05_software\train\rc_mjlab`
  （`v0.6.0` 的 `best` 比赛训练架构，加上 `v0.7.0`～`v0.8.1` 的训练侧改动）。
- 已移植：`src/robot/`（任务注册、环境配置、MDP、地形、HIM 代码）、
  `mjlab/`（固定基准 + 本地补丁的框架）、`pyproject.toml`、`uv.lock`。
- 未移植：`mjcf/`（旧机器人网格 40 MB）、`model_rough.pt`、`model_6800.onnx`
  （旧机器人权重，与本工程观测/动作顺序不兼容）、`sim2sim/`、`mujoco_sim/`、
  `tools/nav_tools/`（部署与工具链，不属于训练算法）。
- 机器人模型：`go2w/`（Unitree go2w，未做任何修改）。

## 2. 名称映射

| 旧 16DOF 机器人 | go2w | 说明 |
| --- | --- | --- |
| `fl_hip_abduction_joint` | `FL_hip_joint` | 外展，轴 `x`，±1.0472 rad |
| `fl_hip_pitch_joint` | `FL_thigh_joint` | 髋俯仰，轴 `y` |
| `fl_knee_joint` | `FL_calf_joint` | 膝，轴 `y`，`-2.7227 ~ -0.83776` |
| `fl_wheel_joint` | `FL_wheel_joint` | 轮，轴 `y`，连续 |
| `fl_wheel_Link` | `FL_wheel_link` | 轮体（接触副本体） |
| `base_link` | `base_link` | 相同 |
| 腿序 `fl, fr, rl, rr` | `FL, FR, RL, RR` | go2w 用大写前缀 |

所有名称模式集中在 `src/robot/naming.py`，`env_cfgs.py` / `rewards.py` /
`robot_cfg.py` 都从那里取值。再换机器人时只需要改这一个文件
（必要时加 `robot_cfg.py` 的执行器与位姿）。

## 3. 从 go2w 模型实测/推导的数值

| 量 | 值 | 来源 |
| --- | --- | --- |
| 轮半径 | `0.086 m` | 名义站姿下轮心到轮网格最低点的距离 |
| 轮距 | `0.385 m` | 左右轮心 `y = ±0.1924` |
| 站姿 | hip `0`、thigh `0.9`、calf `-1.8` | 关节限位内，轮子正好落地 |
| 机身高度 | `0.36 m` | 正运动学 0.359 m |
| 匍匐姿 | hip `±0.4`、thigh `1.65`、calf `-2.55` | calf 下限 `-2.72` 内 |
| 匍匐高度 | `0.20 m` | 正运动学 0.196 m |
| hip/thigh 力矩上限 | `23.7 Nm` | `go2w.xml` 的 `<motor ctrlrange>` |
| calf 力矩上限 | `45.43 Nm` | 同上 |
| wheel 力矩上限 | `15 Nm` | 同上 |

这些都可以用下面这条命令重新验证（只需要 `mujoco`，不需要 GPU）：

```bash
uv run --no-project --with mujoco python tools/check_go2w_contract.py
```

### 端到端自检

装好环境后（CPU 也能跑，mjlab 会用 mujoco-warp 的 CPU 后端）：

```bash
uv run python tools/smoke_build_env.py
```

本机实测（无 NVIDIA GPU，CPU 后端）三个任务都能 build / reset / step：

| 任务 | 动作 | actor obs | critic obs | 奖励项 | 10 步 |
| --- | --- | --- | --- | --- | --- |
| `Robot-Flat-v0` | 16 | 53 | 333 | 16 | 正常，奖励有限 |
| `Robot-Rough-v0` | 16 | 53 | 333 | 24 | 正常，奖励有限 |
| `Robot-Crawl-v0` | 16 | 53 | 333 | 15 | 正常，奖励有限 |

动作顺序实测为
`[FL_hip, FL_thigh, FL_calf, FR_…, RL_…, RR_calf] ++ [FL_wheel, FR_wheel, RL_wheel, RR_wheel]`，
和上面的约定一致。critic 333 = 53 + 线速度 3 + 轮地接触 4 + 高度扫描 273（21×13）。

## 4. 关节/动作/观测顺序

mjlab 的 `SceneEntityCfg` 默认按 **MJCF 顺序**解析名称，go2w 的顺序是：

```text
FL_hip, FL_thigh, FL_calf, FL_wheel, FR_hip, ..., RR_calf, RR_wheel
```

- 观测 `joint_pos` / `joint_vel`：12 个腿关节 → `FL(hip,thigh,calf), FR, RL, RR`；
  轮速 4 维 → `FL, FR, RL, RR`。
- 动作 16 维：`[12 个腿关节位置] ++ [4 个轮速]`（两个 action term 顺序拼接）。
- `src/robot/rsl_rl/wrappers/him_mjlab_vec_env_wrapper.py` 里的
  `reorder_indices` 已从旧的「关节类型优先」改成 go2w 的顺序
  `[0,1,2,4,5,6,8,9,10,12,13,14,3,7,11,15]`（策略侧仍是 `FL,FR,RL,RR` 逐腿排列）。

> 因为顺序与关节名都变了，旧机器人的 `model_rough.pt` / `model_6800.onnx`
> **不能**直接加载到本工程。

## 5. 与源工程的行为差异

1. **轮速指令符号**（`mdp/rewards.py: wheel_roll_tracking`）
   新增 `side_signs` 参数。旧机器人左右轮轴方向不同，符号是 `[+,-,+,-]`；
   go2w 四个轮关节都绕 `+y` 旋转，偏航项必须是 `[-1,+1,-1,+1]`
   （左轮慢、右轮快才是正偏航）。`env_cfgs.py` 传入 `naming.WHEEL_SIDE_SIGNS`。
2. **碰撞**
   旧工程用 `CollisionCfg(contype=0, conaffinity=1, ...)` 全局关闭自碰撞，
   并给轮子覆盖 `condim=6 / priority=1 / friction`。go2w 的 MJCF 已经自带
   （轮子 `class="foot"`：`condim=6, priority=1, friction=0.8 0.02 0.01`），
   所以这里不再覆盖，**自碰撞保持开启**。若训练变慢或出现自碰抖振，
   可在 `robot_cfg.py` 按注释恢复旧的 `CollisionCfg`。
3. **力矩上限**
   旧工程统一 `17 Nm`；本工程按 go2w 电机型号分三档。
4. **几何与位姿**
   轮半径 / 轮距 / 站高 / 匍匐高度 / 低杆净空都按 go2w 改过
   （低杆 `0.28~0.40 m`，因为 go2w 匍匐时机身顶部约 `0.26 m`）。
5. **`leg_symmetry`**（当前配置未启用）
   关节名查找从 `pitch/knee` 扩展为同时接受 `thigh/calf`。
6. **模型加载方式**
   `robot_cfg.get_spec()` 不再用 `MjSpec.from_file(绝对路径)`，而是读 XML 文本 +
   网格字节后 `MjSpec.from_string(xml, assets=...)`。原因是 **MuJoCo 在 Windows
   上无法打开含非 ASCII 字符的路径**，而本工程位于 `D:\桌面\...`。
   同样原因，`mjlab/src/mjlab/scene/scene.py` 也加了一处兼容读取（见第 6 节）。
7. **spec 上的两处清理**（`robot_cfg.get_spec()`）
   - go2w 把轮子视觉网格放在 geom group 0，而 mjlab 生成的地形也在 group 0、
     critic 的 height scanner 只对 group 0 投射射线，会让扫描打到自己的轮子上。
     这些网格被移到视觉组 2（物理不变）。
   - go2w 给所有 geom 设了 `margin=0.001`，而 mujoco-warp 在 MULTICCD
     （`SimulationCfg` 里的 `ccd_iterations=80`）下拒绝非零 margin 的 geom 对，
     直接报 `NotImplementedError`。这里把 margin 清零，保持仿真器的求解器设置
     不变。
8. **日志后端**
   mjlab 1.4 的 `RslRlBaseRunnerCfg.logger` 默认是 `wandb`，没有 `wandb login`
   时 `uv run train` 会直接报 `No API key configured`。`rl_cfg.py` 里改成
   `logger="tensorboard"`（本地 `logs/rsl_rl/...`），需要时用
   `--agent.logger wandb` 切回。PPO 超参没有改动。

以下部分**保持原样**：奖励权重与结构、命令采样与课程、域随机化、PPO 超参
（`config/rl_cfg.py`）、观测缩放与噪声、低通/延迟动作、地形生成器、`mjlab` 本地补丁。

> 唯一的结构性改动是正奖励截断：原来是**全局打补丁**（`enable_only_positive_rewards()`
> 直接替换 `RewardManager.compute`，哪个任务最后构建就影响整个进程），现在改成
> **按环境配置的开关** `cfg.only_positive_rewards`（Rough/Crawl 为 `True`，
> Flat/StepStone 为 `False`），语义不变但不再跨任务泄漏。

## 6. 依赖修复

归档的 `uv.lock` 把 MuJoCo 锁到了 `3.8.1.dev907589621`，该 nightly 已被
`py.mujoco.org` 删除（404），直接 `uv sync` 会失败。处理：

- `pyproject.toml` 的 `override-dependencies` 从 `mujoco>=3.8.0.dev0` 改为
  `mujoco>=3.8.0,<3.9`；
- `mujoco` 不再从 `py.mujoco.org` 解析，改用默认源里的稳定版 **3.8.1**；
- `uv.lock` 已重新生成。

其余版本（torch cu128、warp-lang、mujoco-warp 的 git rev、rsl-rl 5.4.0）未动。

`mjlab` 源码另加了 2 处本地补丁（详见 `DEPENDENCIES.md`）：

1. `envs/mdp/dr/actuator.py`：源工程自带，保留；
2. `scene/scene.py`：`scene.xml` 读取增加非 ASCII 路径兼容（`_load_scene_spec`），
   否则本工程目录下的任何场景都建不起来。

## 7. 已知限制 / 待办

- **训练必须有 NVIDIA GPU**：`uv run train` 走 mjlab 的 GPU 启动器（`select_gpus`），
  没有 GPU 会直接 `IndexError`。mjlab 的 mujoco-warp 有 CPU 后端，所以
  `tools/smoke_build_env.py` / `tools/smoke_train_ppo.py` 能在 CPU 上跑通
  build + step + 一次 PPO 迭代（首次要编译内核，几分钟），但 2048 环境的正式
  训练在 CPU 上不现实。`uv run play` 同样需要 GPU。
- **HIM 训练路径不可用**：`src/robot/rsl_rl/` 依赖外部包 `him_mjlab`
  （`HimManagerBasedRlEnv`），源工程归档里没有该包，`uv.lock` 里也没有。
  当前真正可跑的是 mjlab 自带的 PPO + 非对称 Actor-Critic（`rl_cfg.py`）。
- **地图尚未接入**：`Robocon2027.xml` 里的 **stepping_stones（梅花桩）段** 已经
  参数化接进训练（见 `docs/STEPSTONE.md` 与下面的第 8 节），其余段（高墙、低杆、
  楼梯、轮胎阵等）仍只是地图文件，没有参与训练。
  要接入固定赛道时大概率有两种做法：
  1. 作为一个 `TerrainEntityCfg` 的固定地形（把 XML 展开成 spec 后与机器人
     场景合并，适合 `terrain_type="plane"/"custom"` 的固定赛道）；
  2. 用它替换 `SceneCfg.terrain` 的来源，或在 `MjSpec` 层面 include，
     让 `ContactMatch(pattern="terrain")` 仍然能匹配到地形 body。
  两种方式都要保证地形实体/body 名叫 `terrain`，否则接触传感器
  （`feet_ground_contact`、`base_ground_contact`、`body_collision`）会失配。
- **`go2w/scene.xml`、`go2w/scene_terrain.xml`** 是官方场景，未参与训练；
  训练时的平地/地形由 mjlab 的 `TerrainGeneratorCfg` 程序化生成。

## 8. 梅花桩任务（新增内容，不属于移植）

按 `16DOF_RL.md` 的训练规划与 `Robocon2027.xml` 实场参数新增，**不影响** Flat /
Rough / Crawl 三个已有任务的任何参数：

| 文件 | 作用 |
| --- | --- |
| `src/robot/terrains/stepping_stones.py` | 两条交错桩列地形，桩高/中心距/错位/噪声/缺桩随难度变化 |
| `src/robot/mdp/stepstone.py` | 前进进度、轮滑、机身高度带、到达终点、跌落、地形课程 |
| `src/robot/config/env_cfgs.py::stepstone_env_cfg` | 新任务的观测/指令/奖励/终止/课程 |
| `src/robot/__init__.py` | 注册 `Robot-StepStone-v0` |
| `docs/STEPSTONE.md` | 地形/课程/奖励/终止的完整说明与讨论稿对照 |

要点：

- 实场桩 0.20 × 0.20 × 0.20 m、两列横向间距 0.40 m（≈轮距）、同列中心距 0.60 m、
  错位 0.30 m；训练地形以这些值为难度 1.0，从“桩高 0.06 m / 中心距 0.30 m”逐步升上来。
- 桩区默认 0.6 m 深坑，掉坑 = 失败，防止策略从地面绕过。
- Actor 观测从 53 维扩到 **326** 维（+273 维局部高度扫描），其余任务不变。
- 地形课程按“是否通过整段地形”升降级；速度指令按跟踪率自适应放宽。
- 尚未实现：真实 LiDAR 噪声/丢点/遮挡、几何可达性检查、逐 episode 地形种子日志
  （对应讨论稿第 2.4、4.4 节，属于后续感知阶段）。
