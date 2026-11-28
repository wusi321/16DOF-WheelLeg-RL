# 单木桥（SingleBeam）—— 16DOF 轮腿机器人

> 场地资料（公共）｜ 2026-11-28
> 状态：**地形按官方规格新建完成、任务已注册，还没训练过**

---

## 1. 这是什么场地

一条窄长的木头横在场地中间，机器人要从一头走到另一头。

**官方规格（《2027 年足式机器人挑战赛 V1.0》规则 3.1 节）原文**：

> 单木桥长 3000mm，宽 400mm，高 100mm。表面涂绿色水性漆。
> 称 400mm 为短边，3000mm 为长边。

**通过判定（规则 4.2 节）**：

> 机器人从一侧短边进入单木桥，并从另一端短边离开，则表示障碍通过完成。
> 通过单木桥时，机器人从障碍物侧面摔下，或 1/2 以上足与地面接触（含 1/2），
> 则需重新通过。

| 项目 | 规格 |
|---|---|
| 长 | 3000 mm |
| 宽 | 400 mm |
| 高 | 100 mm（离地） |
| 护栏 | **无** |
| 失败条件 | 从侧面摔下，或四条腿里两条着地（含两条） |

---

## 2. 这个场地特殊在哪

**它是全场唯一要求"持续走直线"的障碍。**

- 梅花桩：踩一下、跨一步、再踩一下，中间可以调整
- 单木桥：**连续 3 米，宽度只有 400mm**，两侧没有任何护栏，一旦偏出去就掉下来

所以它练的是**横向稳定 + 持续纠偏**，这两种能力在别的场地都练不到。

`400mm` 这个宽度值得注意：机器人轮距是 385mm，也就是说**左右轮之间的距离和桥宽几乎一样**——
容错空间只有 15mm，基本等于要求"四轮必须始终在桥面上"。

---

## 3. 训练命令

任务名：**`Robot-16DOF-SingleBeam-v0`**（2026-11-28 新注册）

```powershell
$root="D:\Project\nano\16dof-rl"; $py="D:\Project\wheelleg-rl\.venv\Scripts\python.exe"
$env:PYTHONPATH="$root\src;$root\mjlab\src"; $env:PYTHONIOENCODING="utf-8"
$env:WARP_CACHE_PATH="$root\.cache\warp"; $env:TMP="D:\Project\nano\.tmp"; $env:TEMP=$env:TMP
Set-Location $root

& $py -u train_16dof.py Robot-16DOF-SingleBeam-v0 --agent.max-iterations 1200 --agent.run-name "beam_第一轮"
```

---

## 4. 难度怎么设计的

规则尺寸是固定的（400mm 宽不能改），所以课程从两个方向收紧：

| 难度 | 桥宽 | 桥高 | 说明 |
|---|---|---|---|
| 0 | 400 mm | 60 mm | 矮一点，轮子容易爬上去 |
| 0.5 | 360 mm | 80 mm | |
| 1 | **320 mm** | **100 mm** | 比官方 400mm 还窄，等于给比赛留余量 |

**为什么训练要比比赛更窄**：训练时练 320mm 都能走稳，比赛 400mm 就有 80mm 的冗余。
这和梅花桩那边"落点要比要求更准"是同一个思路。

---

## 5. 判据为什么只写"前进 + 到终点"

没加任何"保持平衡"或"别掉下去"的特殊奖励，理由：

1. **掉下去就是踩到地面**，会触发 `base_too_low` 终止（阈值 0.10 m，比桥面 0.10 m 略低，
   所以站在桥上不会被误判）。回合直接结束，前进奖励自然归零。
2. 物理约束已经很严格了：偏出去就掉，没有第二个结果。再加奖励容易教出怪招
   （比如学会"用两条腿挂在桥边蹭过去"这种钻空子行为）。

---

## 6. 第一次训练要盯什么

1. **最大的风险：它可能学会"不走桥上、直接走地面"**。因为桥只有 400mm 宽，
   而场地有 1600mm 宽，两侧都是平地。如果发现它绕开桥走旁边，说明还缺一个
   "必须走桥面"的约束——这时需要加"横向偏离惩罚"（离开桥中线就扣分）。
2. **可能卡在爬不上去**。100mm 高的坎，轮子（直径 108mm）理论上能爬，
   但如果策略不会抬腿，会一直撞桥边。看 `Curriculum/terrain_levels` 是不是卡住不动。
3. **和梅花桩的关系**：如果梅花桩已经练好了落点精度，跨到单木桥会容易些，
   但单木桥额外要求"持续"，这是梅花桩没有的。

---

## 7. 怎么看这个场地（3D 预览）

仓库根目录有个 `preview_terrain.py`，本文件夹里有个 `查看场地.bat`：

**最简单的办法**：双击本文件夹里的 `查看场地.bat`，会弹出一个 3D 窗口。

命令行方式：

```powershell
cd D:\Project\nano\16dof-rl
$env:PYTHONPATH="D:\Project\nano\16dof-rl\src;D:\Project\nano\16dof-rl\mjlab\src"
python preview_terrain.py 单木桥              # 看比赛难度
python preview_terrain.py 单木桥 --level 0    # 看最简单难度（60mm 高、400mm 宽）
python preview_terrain.py 单木桥 --levels 4   # 四个难度并排对比
```

窗口操作：左键旋转、右键平移、滚轮缩放、Esc 关闭。

---

## 8. 文件清单

```
单木桥/场地/
├── README.md                       ← 本文档
├── 查看场地.bat                    ← 双击打开 3D 预览
├── src/robot/terrains/
│   └── single_beam.py              ← 地形定义（2026-11-28 新写，按官方规格）
└── config/
    └── env_cfg_beam.py             ← 训练配置
```

**集成方法**（三处都要改）：

1. `single_beam.py` 复制到 `src/robot/terrains/`
2. `src/robot/terrains/__init__.py` 里加 `from .single_beam import BoxSingleBeamTerrainCfg`，
   并加进 `__all__`
3. `env_cfgs.py` 顶部 import 加 `BoxSingleBeamTerrainCfg`，再粘入 `env_cfg_beam.py` 的内容

任务注册已在 `task_16dof.py` 的 `register()` 里加好。