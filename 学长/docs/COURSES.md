# 两条新赛道：沟壑课程 & 绕桩课程

除梅花桩外，仓库里还有两条针对"足式障碍赛"的新赛道。两者都复用同一套机器人、
观测（53 + 273 高度扫描）、奖励基线和终止条件，只替换地形与任务奖励。

| 任务 ID | 地形 | 要练的能力 | 曲线图 |
| --- | --- | --- | --- |
| `Robot-GapCourse-v0` | 40 cm 平台 + 5 → 45 cm 的沟 | **跃过沟壑**（不是绕、不是掉） | [gapcourse_levels.png](images/gapcourse_levels.png) |
| `Robot-Slalom-v0` | 一排 50×50 mm 方柱 + 左右交替必经点 | **S 形有序绕桩** | [slalom_levels.png](images/slalom_levels.png) |
| **`Robot-ObstacleCombo-v0`** | 两列并排：一列沟壑、一列绕桩，同场训练 | 一次训练两条赛道 | [combo_levels.png](images/combo_levels.png) |

训练前先出图自查（不占 GPU）：

```bash
uv run python tools/plot_course_terrain.py      # 生成上面两张图
```

---

## 1. 沟壑课程 `Robot-GapCourse-v0`

实现：`src/robot/terrains/gap_course.py` + `gapcourse_env_cfg`。

### 布局

```
[起点平台 0.8 m][0.4 平台][沟][0.4 平台][沟]... [终点平台 0.8 m]
```

- 平台长 **0.40 m**、与起点同高（`slab_height = 0`）；
- 沟宽随难度：**0.05 → 0.45 m**（`_GAP` 线性插值：0.05 / 0.18 / 0.28 / 0.38 / 0.45）；
- 沟深 0.6 m：掉下去触发 `base_too_low` / `base_ground_contact`，回合结束；
- 路程 6.8 m，回合 25 s。

### 奖励（与梅花桩的关键差异）

| 项 | 权重 | 说明 |
| --- | --- | --- |
| `forward_progress` | 1.5 | 净前进 |
| `track_lin_vel_x` / `track_ang_vel_z` | 0.6 / 0.3 | 速度与偏航跟踪 |
| **`leap`（新增）** | **+5** | **四轮离地 ≥2 步后落地 → 记一次"跃沟"** |
| `goal_reached` | +80 | 到达终点 |
| `time_out` | −10 × (1 − dx/6.8) | 磨时间按剩余距离罚 |
| `idle`（新增） | −0.5/步 | 赛道内 75 步没有 3 cm 前进（防原地蹭分） |
| `is_terminated` | −10 | 摔落 |
| `overspeed` | −0.5 | 超过指令 +0.4 m/s |

> **注意：这条赛道不加 `airborne` 惩罚。** 梅花桩里滞空是作弊，这里滞空就是任务本身。

> ⚠️ **跃沟奖励必须"真的过去"**：最初只要"离地 ≥2 步后落地"就给 +5，
> 结果 play 里策略**原地小幅弹跳**刷分、一步不走（2026-10-30）。
> 现在要求**落点比起跳点前进 ≥ 0.25 m**，且起跳/落点都在赛道范围内；
> 另加通用 `idle` 惩罚（赛道内 75 步没有 3 cm 前进 → −0.5/步）。
> 回归检查：`uv run python tools/check_gap_leap.py`
> （原地跳 0 次、微移 0.1 m 0 次、真跃 0.35 m 1 次）。

`leap` 的实现（`src/robot/mdp/gap_course.py`）只看接触状态，不需要知道沟在哪儿：

```
四轮离地计数 ≥ 2  →  重新接触地面  →  +5，计数清零
```

配套 metric `gaps_crossed` 记录每回合跃沟次数，在 wandb 上直接看它有没有在涨。

### 难度与验收

| 等级 | 沟宽 | 期望 |
| --- | --- | --- |
| L0–L2 | 0.05–0.18 m | 滚过去即可 |
| L4–L6 | 0.28–0.35 m | 需要抬腿/小跳 |
| L8–L9 | 0.38–0.45 m | 接近 0.55 m 车长的 82%，必须助跑起跳 |

验收：`Episode_Metrics/gaps_crossed` 稳定 > 0、`Episode_Termination/base_too_low`→0、
`goal_reached` 稳定。

---

## 2. 绕桩课程 `Robot-Slalom-v0`

实现：`src/robot/terrains/slalom.py` + `src/robot/mdp/slalom.py` + `slalom_env_cfg`。

### 布局

- 一排方柱，截面 **50 × 50 mm**、高 300 mm（对齐足式障碍赛横杆/立柱规格），
  沿中心线 y = 0 排列；
- 柱距随难度缩短：**1.00 → 0.52 m**；
- 每根柱子旁边放一个**必经点**（左右交替、横向 ±0.42 m）；
- 必经点在地形里是**不参与碰撞的小方块**（0.04 × 0.04 × 0.002 m），
  MDP 侧按几何签名把它们读出来并按 x 排序 —— 训练代码不需要知道难度或柱距。

```
        ●waypoint            ●waypoint
   ────────┼──────────────────┼────────►  x
           ▮pillar            ▮pillar
       ●waypoint                 ●waypoint
```

### 训练方式（你说的"没想到怎么训"）

核心思路：**把"绕过第 k 根柱子"变成一个显式的、按顺序的里程碑**，而不是让奖励
自己去猜什么是 S 形。

| 项 | 权重 | 什么时候给 |
| --- | --- | --- |
| `waypoint` | **+5** | 按顺序进入下一个必经点半径（0.30 m）内 |
| `all_waypoints` | +30 | 全部必经点走完（一次性） |
| `skip` | **−30** | 吃到了**顺序之外更后面**的必经点（跳点），**且不推进进度** |
| `pillar_hit` | **−5** | 任意轮子贴到柱子（0.5 s 内最多记一次） |
| `goal_reached` | +60 | 到达终点 |
| `forward_progress` | 1.2 | 净前进（防止原地绕圈） |
| `time_out` | −10 × (1 − dx/6.8) | 磨时间 |

反取巧的设计：

1. **必须按顺序**：`idx` 只会 +1；吃到 `idx+1` 之后的点只罚分、不推进，
   所以"跳过两根柱子直接吃第三个点"是纯亏的；
2. **进度指标独立**：`Episode_Metrics/slalom_progress = idx / 总数`，
   reward 涨而它不涨 = 在绕圈/蹭分；
3. **碰柱扣分**：避免"贴着柱子挤过去"这种取巧走法；
4. **难点靠柱距**：柱距 1.00 → 0.52 m，越小转向越急，S 形幅度越大。

### 难度与验收

| 等级 | 柱距 | 期望 |
| --- | --- | --- |
| L0–L2 | 1.00–0.80 m | 轻松绕，主要学"按顺序" |
| L4–L6 | 0.72–0.62 m | 需要连续换向 |
| L8–L9 | 0.56–0.52 m | 急转 S，容易蹭柱 |

验收：`slalom_progress` 单调上升且每回合能到 1.0、`Episode_Reward/skip` 接近 0
（说明没有跳点）、`pillar_hit` 很小。

---

## 3. 怎么跑

```bash
# 先看图
uv run python tools/plot_course_terrain.py

# 训练（20480 并行；8 m 地形下 40960 会 OOM）
uv run train Robot-GapCourse-v0 \
  --env.scene.num-envs 20480 --env.sim.mujoco.ccd-iterations 16 \
  --agent.algorithm.num-mini-batches 32 --agent.save-interval 25 \
  --agent.max-iterations 600 --agent.logger wandb --agent.wandb-project 16dof-wheelleg

uv run train Robot-Slalom-v0 \
  --env.scene.num-envs 20480 --env.sim.mujoco.ccd-iterations 16 \
  --agent.algorithm.num-mini-batches 32 --agent.save-interval 25 \
  --agent.max-iterations 600 --agent.logger wandb --agent.wandb-project 16dof-wheelleg

# 回放（等级采样由环境变量控制，默认 0..9 级各一个）
SLALOM_PLAY_MAX_LEVEL=9 uv run play Robot-Slalom-v0 --checkpoint-file <ckpt> --num-envs 10
GAPCOURSE_PLAY_MAX_LEVEL=9 uv run play Robot-GapCourse-v0 --checkpoint-file <ckpt> --num-envs 10
```

## 4. 已知的调参点

| 想调什么 | 改哪里 |
| --- | --- |
| 沟更宽/更窄 | `BoxGapCourseTerrainCfg` 的 `_GAP`（或 `gap_range` 仅文档） |
| 平台更长 | `slab_length`（默认 0.4 m） |
| 柱子更细/更粗 | `pillar_size`（默认 0.05 m 见方） |
| 柱距更密 | `BoxSlalomTerrainCfg` 的 `_SPACING` |
| 必经点更靠外（S 更夸张） | `waypoint_offset`（默认 0.42 m） |
| 必经点判定太松/太严 | `src/robot/mdp/slalom.py` 的 `WAYPOINT_RADIUS`（默认 0.30 m） |
| 跳点罚太轻 | `skip_penalty` 的 `scale`（默认 −30） |

改完地形记得先 `uv run python tools/plot_course_terrain.py` 出图确认。

---

## 5. 组合地图 `Robot-ObstacleCombo-v0`（推荐先跑这个）

把两条赛道**并排**放进同一张地图：mjlab 的 curriculum 模式给每个 `sub_terrain`
分配**一列**，`proportion=0.5` 让两类环境各占一半；环境按 `terrain_types` 落到
自己的那一列，所以一次训练里同时有 50% 在过沟、50% 在绕桩。

```
        y
     +1.6 ┌────────────── 绕桩列 ──────────────┐   ← terrain_types = 1
          │  ▮ ●   ▮   ●   ▮   ●   ▮          │
       0 ─┼────────────── 分界 ────────────────┤
          │  ███ ███ ███ ███ ███ ███ ███       │   ← terrain_types = 0
     -1.6 └────────────── 沟壑列 ──────────────┘
          0 ──────────────── x ──────────────→ 8 m
```

### 奖励怎么按赛道分发

`src/robot/mdp/combo.py` 用 `terrain_types` 做掩码，每个奖励项只对自己那条赛道生效：

| 奖励 | 只对哪列生效 |
| --- | --- |
| `leap`（+5）、`gaps_crossed` 指标 | 沟壑列 |
| `waypoint` / `all_waypoints` / `skip` / `pillar_hit`、`slalom_progress` 指标 | 绕桩列 |
| 前进 / 速度跟踪 / 姿态 / `goal_reached` / `time_out` / `is_terminated` | 两列都生效 |

设计要点：**不掩码就会串味**——沟壑的环境如果也能吃到绕桩的必经点奖励（或反过来），
两条赛道的信号会互相污染。实测掩码后：沟壑环境 `slalom waypoints = 0`、
绕桩环境 = 6（L0），各拿各的。

### 用法

```bash
# 先看图（上=绕桩列，下=沟壑列）
uv run python tools/plot_course_terrain.py

# 训练
uv run train Robot-ObstacleCombo-v0 \
  --env.scene.num-envs 20480 --env.sim.mujoco.ccd-iterations 16 \
  --agent.algorithm.num-mini-batches 32 --agent.save-interval 25 \
  --agent.max-iterations 400 --agent.logger wandb --agent.wandb-project 16dof-wheelleg

# 回放
COMBO_PLAY_MAX_LEVEL=9 uv run play Robot-ObstacleCombo-v0 --checkpoint-file <ckpt> --num-envs 10
```

### 看什么

| 指标 | 期望 |
| --- | --- |
| `Episode_Metrics/gaps_crossed` | 只有沟壑列有值，随训练上升 |
| `Episode_Metrics/slalom_progress` | 只有绕桩列有值，收敛到 1.0 |
| `Episode_Reward/leap` vs `waypoint` | 两条曲线应同时增长；只有一条在动 = 掩码错了 |
| `Episode_Termination/base_too_low` | 应压到 0（沟壑那边掉沟会顶起来） |

> 冒烟已验证：6 → 7 个任务全部 build/reset/step 通过，零动作站立 100 步不摔。