# 02 · wandb 数据监控：怎么读曲线，怎么判断"是不是崩了"

本项目的训练日志都推到 wandb（`--agent.logger wandb`）。这篇讲三件事：
**怎么用**、**面板怎么摆**、**看到怪曲线怎么诊断**——全部用本项目的真实曲线当例子。

---

## 1. 基本用法

```bash
# 第一次用
wandb login                      # 粘贴 API key（只做一次）

# 训练时自动上传
uv run train Robot-StepStone-v0 --agent.logger wandb --agent.wandb-project 16dof-wheelleg

# 断网/集群无外网：先离线记录，事后补传
WANDB_MODE=offline uv run train ...
wandb sync wandb/offline-run-*
```

- 每次训练会在 `logs/rsl_rl/<task>/<启动时间>/` 生成 checkpoint，wandb 的 run 名
  默认就是那个**启动时间**，所以曲线和图、视频、checkpoint 能对上。
- run 页面里的 `Files` 能看到保存的 `.pt`，本地 `logs/.../*.pt` 同名。
- 需要时用 API 拉历史：

```python
import wandb
api = wandb.Api()
run = api.run("scunf1001-shandonghuayu/16dof-wheelleg/xxxxxxx")
for row in run.scan_history(keys=["Train/mean_reward", "Curriculum/terrain_levels/mean"]):
    ...
```

---

## 2. 面板怎么摆（本项目的分组）

mjlab 的 key 自带前缀，直接按前缀分组即可：

| 分组 | 关键 key | 用途 |
| --- | --- | --- |
| **Train** | `Train/mean_reward`、`Train/mean_episode_length` | 一眼看整体趋势 |
| **Curriculum** | `Curriculum/terrain_levels/{mean,max,upgraded,downgraded,level_cap}`、`Curriculum/command_x_levels/x_range_max` | 课程是否在推进、是否卡死/翻烧饼 |
| **Episode_Metrics** | `Episode_Metrics/stone_coverage`、`wheel_slip`、`goal_reached` | **任务是否真的完成**（比 reward 可信） |
| **Episode_Reward** | 各奖励项 | 哪一项在主导、哪一项在恶化 |
| **Episode_Termination** | `base_ground_contact`、`base_too_low`、`bad_orientation`、`time_out`、`goal_reached` | 回合怎么结束的 |
| **Loss / Policy** | `Loss/value`、`Loss/surrogate`、`Loss/entropy`、`Loss/learning_rate`、`Policy/mean_std` | PPO 本身是否健康 |

**建议两行布局**：

```
第一行（判断成败）：mean_reward | mean_episode_length | terrain_levels/mean | stone_coverage
第二行（定位原因）：Episode_Termination/* | flat_orientation | forward_progress | Loss/*
```

---

## 3. 曲线怎么读：四个"主干指标"

按顺序看，任何一条不对就停下来：

### ① `Train/mean_episode_length`

它比 reward 更早暴露问题。

| 形态 | 含义 |
| --- | --- |
| 快速上升后稳定 | 正常（学会不摔了） |
| **冲到超时上限（如 900 步 = 30 s）** | **在磨时间**（案例 B） |
| **掉到很短（如 58 步）** | **极端策略**：要么飞越（案例 A），要么"冲线即止" |
| 长期在小值徘徊 | 频繁摔，还没学会 |

### ② `Episode_Termination/*`

看**回合是怎么结束的**，比 reward 直接。

- `goal_reached` 占比高 → 好；
- `time_out` 占比涨 → 磨时间；
- `base_ground_contact` / `base_too_low` / `bad_orientation` 尖峰 → 摔；
  如果尖峰只出现在**升级难度的瞬间**，是正常的适应期。

### ③ `Curriculum/terrain_levels/*`

<img src="../../assets/images/2026-10-26_22-26-00/1.png" width="760">

> 上图是第 1 轮的课程面板：`level_cap` 每约 50 迭代放行一级（左下）、
> `mean/max` 阶梯上升（中下/中上）、`downgraded` 在 50 迭代后归零（右下）。
> **课程面板要横着读三张图**：cap（允许到哪）、mean（实际到哪）、upgraded/downgraded（流动性）。

- `mean` 平滑上升、`max` 逼近上限、`upgraded` 稳定在 0.3–0.6 → 健康；
- `mean` 长期不动 + `upgraded`≈0 + `downgraded`≈0 → **课程死区**（卡在某一级，
  既不升也不降）；
- `upgraded` 与 `downgraded` **反相方波**（0↔1 跳）→ **整批同步翻转**
  （本项目第 2 轮、`2026-10-27_23-24-31` 都是）；
- `level_cap` 上升但 `mean` 不跟 → 难度跑得比能力快。

### ④ `Episode_Metrics/stone_coverage`（任务自己的硬指标）

**这是本项目最可信的指标**：

- 从 0 涨到 0.8+ → 真的在踩桩；
- **卡在某个值不动** → 找到了"平推"或"只踩一条列"的捷径（案例 C、第 3 轮的 0.50）；
- 涨但 `goal_reached` 归零 → 在摸桩不冲线（案例 B）。

---

## 4. "是不是崩了" 的判断流程

```
损失/熵/lr 正常，只有任务指标崩  → 奖励结构被绕过（最常见）
损失爆炸（value loss 10x+）       → 数值/尺度问题：检查权重、clip、nan
指标全 0 或全 NaN                 → 环境 bug：观测没更新、终止条件写错
reward 涨但 termination 变差      → 局部最优（飞越/磨时间/平推）
reward 与 coverage 同时涨又同时崩  → 课程推太快
```

### 4.1 三种"看起来崩了"其实是正常现象

| 现象 | 什么时候正常 | 怎么区分 |
| --- | --- | --- |
| **升级难度时的 `Loss/value` 尖峰** | 分布变化，随难度阶梯出现 | 尖峰位置与 `level_cap` 台阶对齐就是正常 |
| **`mean_reward` 台阶式下跌** | 同上，新难度更贵 | 看 `Episode_Termination` 是否同步变差；恢复一般 <20 迭代 |
| **`entropy` 单调下降** | 策略在收敛 | 降到 10 以下且动作开始变粗暴时要注意（本项目的"点头撅屁股"） |

### 4.2 三种"看着还行其实崩了"的陷阱

| 曲线 | 陷阱 | 本项目实例 |
| --- | --- | --- |
| `mean_reward` **上涨** | 可能是靠捷径涨的 | 第 2 轮 10→37，同时 `upgraded` → 0.05 |
| `stone_coverage` **高** | 可能只覆盖一条桩列 | 第 3 轮恒定 0.50（旧双列地形） |
| `goal_reached` **稳定** | 可能只靠 2.4 m/s 冲过去的 | 第 1 轮稳定，但 `track_lin_vel_x`≈0 |

> 结论：**永远不要只看 reward**。项目要有一项"规则级"指标（本项目的
> `stone_coverage`），它才是验收标准。

---

## 5. 三次训练的诊断实例（带截图）

### 第 1 轮 `2026-10-26_22-26-00`：飞越

<img src="../../assets/images/2026-10-26_22-26-00/8.png" width="760">

| 曲线 | 值 | 怎么读 |
| --- | --- | --- |
| `Train/mean_reward`（左下） | 爬到 8.5 走平 | 看起来"训练成功" |
| `Train/mean_episode_length`（右下） | 300 → **58 步** | ⚠️ 真正的报警：1.16 s 走完 2.8 m = 2.4 m/s |
| `Policy/mean_std`（上） | 0.95 → 0.45 | 策略收敛（收敛到一个坏行为） |

<img src="../../assets/images/2026-10-26_22-26-00/3.png" width="760">

- `stone_coverage`（中上）**只有 0.43**，且从 150 迭代起就开始下降；
- `wheel_slip`（左上）涨到 1.3：轮子在桩上疯狂空转——它在"起跳前挠地"；
- `goal_reached`（中下）稳定 0.05：成功终止很多，但都是"飞过去"的成功。

判定：**奖励被绕过**（终点奖励与覆盖率无关 + 进度项无上限）。修复 `c64bb45`。

### 第 2 轮 `2026-10-27_07-56-35`：摆烂

<img src="../../assets/images/2026-10-27_07-56-35/10.png" width="760">

- 560 迭代：`mean_reward` 10 → **37**，`mean_episode_length` 200 → **900 步**；
- 这两个**同时**出现就是"磨时间"的确诊信号（reward 涨、回合顶到超时上限）。

<img src="../../assets/images/2026-10-27_07-56-35/1.png" width="760">

- `upgraded`（左上）从 0.95 掉到 **0.05**、`distance`（右下）掉到 1.0–1.5 m：
  它不再尝试过终点，只在桩上磨；
- 同一轮的 `capped` 面板（`2026-10-27_07-56-35/2.png`）早期一直顶在 1：
  课程封顶在钳制，说明难度推进得比能力快。

<img src="../../assets/images/2026-10-27_07-56-35/3.png" width="760">

- `stone_coverage`（中上）反而涨到 **0.85**、`goal_reached`（中下）→ **归零**；
- 这是最容易被骗的一张图：**覆盖率很高、奖励很高、任务失败**。

判定：**局部最优**（摸桩 + 等超时）。修复 `6843e16`（终点 120、超时罚、单桩 2）。

### 第 3 轮 `2026-10-28_10-23-47`：稳定通过 + 末期退化

<img src="../../assets/images/2026-10-28_10-23-47/8.png" width="760">

- `base_ground_contact`（中上）、`bad_orientation`（右上）在 30 迭代后归零；
- `base_too_low`（左上）在 300–370 有一波（≈12 次/迭代），随后压到 ≈0：
  这是 `level_cap` 推到 5–6 级的**适应期**，属于正常，不是崩；
- 下面一排 `Loss/*`：value loss 在 0.01–0.03、surrogate 平稳、lr 单调下降——**PPO 健康**。

<img src="../../assets/images/2026-10-28_10-23-47/3.png" width="760">

- `stone_coverage` 60 迭代冲到 **0.50** 后 700 多个迭代不动（旧双列地形，只踩一条列）；
- `mean_leg_action_acc` 1.9 → 0.9：动作越来越柔——因为它不需要抬腿；
- `wheel_slip` 200 迭代见顶后下降：不再打滑，但也不是因为踩得准。

<img src="../../assets/images/2026-10-28_10-23-47/6.png" width="760">

- 这就是"点头撅屁股"的证据：`flat_orientation`（右上）从 500 迭代的 −0.004
  滑到 800 迭代的 **−0.007**，`airborne`（右下）同步变负（−0.005 → −0.006）；
- 同时 `goal_reached` / `forward_progress` / `all_stones` 都**没有再涨**——
  说明后面 100 多个迭代只是在"加大幅度"，没有学到新东西。

判定：**前 650 迭代可用，之后只有"晃"**。交付 `model_650.pt`，并把
`--agent.max-iterations` 固定为 650。

<img src="../../assets/images/2026-10-28_10-23-47/9.png" width="760">

- `Loss/entropy`（上）22.5 → 10，700 迭代后基本贴地：探索停了；
- `Metrics/twist/error_vel_xy`（右下）稳定 0.5、`error_vel_yaw` 0.28：
  速度跟踪其实一直不算好，但任务照样能过——**说明"过障碍"不需要精确跟踪**，
  这也解释了为什么速度课程很早就打满。

### 诊断顺序小结

```
1. Episode_Termination  → 回合怎么结束的？（摔 / 超时 / 成功）
2. mean_episode_length  → 时长是否合理？（过短=飞越，顶格=磨时间）
3. stone_coverage       → 任务是否按规则完成？（卡住=捷径）
4. Episode_Reward/*     → 哪一项在主导、哪一项在恶化（姿态/滑移/滞空）
5. Loss/* + entropy     → 确认 PPO 本身没崩（99% 的情况都是健康的）
```

---

## 6. 小技巧

- **对比两次 run**：wandb 面板上把两个 run 拖到同一张图（如图中绿/粉/蓝三色），
  比看单条曲线直观得多。
- **打完 tag 再走**：`wandb run --tags flyaway`，后面找"那次飞越的"很快。
- **本地留存**：`wandb/` 目录不要提交（`.gitignore` 已排除），但要留一份，
  离线时可以 `wandb sync` 补传。
- **截图归档**：每次训练结束把关键曲线截图存到 `assets/images/<启动时间>/`，
  再在 `docs/TRAINING.md` 里写三行结论——半年后你只会记得这张图。
