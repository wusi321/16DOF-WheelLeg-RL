# 高墙（Wall / 翻墙）—— 16DOF 轮腿机器人

> 负责人：sys ｜ 独立任务配置：公共 ｜ 2026-11-28
> 状态：**地形早就有、任务 2026-11-28 才独立注册，还没单独训练过**

---

## 1. 这是什么场地

机器人要连续翻越 5 道横向挡墙，一路走到终点：

| 项目 | 规格 | 说明 |
|---|---|---|
| 墙 | **0.8 m 宽 × 8 cm 厚 × 最高 0.30 m** | 官方规格（规则表 1）：高 300mm、宽 800mm、厚度 80mm |
| 墙的数量 | 5 道 | 位置在 x = 2.1 / 3.2 / 4.3 / 5.4 / 6.5 m |
| 赛道 | 8.0 × 1.6 m，目标 6.8 m | |
| 回合 | 30 秒 | |

**难度从 0 到 1，墙高从 0 长到 0.30 m**。注意难度 0 时墙高是 **0**——整条赛道是平的。
这是故意的：轮腿机器人不可能一上来就翻 20 cm 墙，先让它在没有墙的赛道上把前进
跑顺，墙再一点点长起来。

> **2026-11-28 修正**：旧版写的是"1.0 m 宽 × 5 cm 厚 × 最高 0.35 m"，三个数全错。
> 官方规则表 1 明确是**高 300mm、宽 800mm、厚 80mm**，已按官方值改正。
> （墙宽只有 800mm，而赛道宽 1.6m，所以两侧会各留 400mm 空隙——这是官方尺寸决定的，
> 不是"留来观察的"。）

---

## 2. 为什么现在才独立成一个任务

这个地形**一直存在**，但之前只作为「粗糙地形」（`Robot-16DOF-Rough-v0`）里
**占比 15%** 的一种地形跟着一起练，旁边还混着台阶、乱石、斜坡。

问题在于：**策略会挑软柿子捏**。一条赛道上如果 85% 的地形都是容易的，它就会
专挑容易的走，翻墙能力永远练不出来。

所以 2026-11-28 把墙单独拆成一个任务：整条赛道只有墙，躲不开。

---

## 3. 怎么训练

任务名：**`Robot-16DOF-Wall-v0`**（2026-11-28 新注册）

```powershell
$root="D:\Project\nano\16dof-rl"; $py="D:\Project\wheelleg-rl\.venv\Scripts\python.exe"
$env:PYTHONPATH="$root\src;$root\mjlab\src"; $env:PYTHONIOENCODING="utf-8"
$env:WARP_CACHE_PATH="$root\.cache\warp"; $env:TMP="D:\Project\nano\.tmp"; $env:TEMP=$env:TMP
Set-Location $root

& $py -u train_16dof.py Robot-16DOF-Wall-v0 --agent.max-iterations 1200 --agent.run-name "wall_第一轮"
```

回放（层级面板拉到 9 看实场难度）：

```powershell
& $py play_16dof.py Robot-16DOF-Wall-v0 --agent trained --num-envs 1 --viewer viser `
    --checkpoint-file "<模型路径>.pt"
```

---

## 4. 判据是怎么设计的（以及为什么）

| 判据 | 权重 | 为什么这么设 |
|---|---|---|
| `forward_progress` | 1.8 | 目标速度 0.9 m/s——翻墙要冲，太慢爬不上去 |
| `track_lin_vel_x` | 0.6 | 跟住速度指令 |
| `leap`（跃过奖励） | 4.0 | 离地 ≥2 步后落地给分。翻墙本质是一次小跳，这个信号告诉它"跳对了" |
| `goal_reached` | 80.0 | 到终点的大奖励 |
| `idle` | −0.5 | 原地磨蹭扣分 |
| `overspeed` | −0.5 | 别冲过头 |

**刻意没做的两件事**：

1. **不加"离地/抬腿"惩罚**。其它场地常用离地罚压制乱抬腿，但这个任务就是要跳起来——
   加了等于惩罚正确动作。跨沟任务也是同样的道理。
2. **不额外加"撞墙惩罚"**。墙是实心的碰撞体，撞上去会把它挡住、走不动，
   前进奖励自然就少了。物理约束本身够了，再加惩罚容易引发"逃避"行为
   （梅花桩那边踩过这个坑：惩罚给重了，它宁可摔倒让回合早点结束）。

---

## 5. 第一次训练要盯什么

因为**没单独训过**，以下是预判的风险：

1. **可能学成"绕过去"**。如果发现它贴着赛道边缘走、绕开墙，说明墙没有横贯整个宽度
   （当前是赛道宽度的 0.8 倍，两侧留了空隙用于观察）。真出现这种情况，
   把 `wall_length_frac` 调到 1.0。
2. **可能学成"慢慢蹭"**。看 `leap` 曲线有没有涨——不涨说明它从没真正跳起来过。
3. **课程推进太快会卡死**。如果 `Curriculum/terrain_levels` 涨到某一级就不动了，
   说明那一级墙太高，需要调低 `wall_height_range` 的上限再训。
4. **翻墙和跨沟的技能可能互通**。跨沟（GapCourse）练的也是"跳过去"，
   如果先训好了跨沟，翻墙可能上手更快，可以试试从跨沟模型接着训。

---

## 6. 文件清单

```
高墙/场地/
├── README.md                          ← 本文档
├── src/robot/terrains/
│   └── competition_terrains.py        ← 三种比赛障碍地形（高墙/低杠/金字塔台阶）
└── config/
    └── env_cfg_wall_and_lowbar.py     ← 高墙与低杠的训练配置（2026-11-28 新写）
```

> `competition_terrains.py` 里其实有三个地形类：
> - `RCWallTerrainCfg` —— 高墙（本文档）
> - `RCLowBarTerrainCfg` —— 低杠（见 `其他/场地/README_lowbar.md`）
> - `RCPyramidStairsTerrainCfg` —— 金字塔台阶（目前只在「粗糙地形」里用过）
---

## 怎么看这个场地

双击本文件夹里的 `查看场地.bat`，会弹出一个 3D 窗口，直接看到场地长什么样。

或者用命令行：

```powershell
cd D:\Project\nano\16dof-rl
$env:PYTHONPATH="D:\Project\nano\16dof-rl\src;D:\Project\nano\16dof-rl\mjlab\src"
python preview_terrain.py 高墙              # 看比赛难度
python preview_terrain.py 高墙 --levels 4   # 四个难度并排对比
```

窗口操作：左键旋转、右键平移、滚轮缩放、Esc 关闭。