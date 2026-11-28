# 限高框架（LowBar）—— 16DOF 轮腿机器人

> 场地资料（公共）｜ 2026-11-28 按官方规则重写
> 状态：**地形已按官方规格修正，任务已注册，还没训练过**

---

## 1. 这是什么场地

一个 800×800 的绿色方框，内空只有 290mm 高。机器人必须以任何姿态从下面钻过去。

**官方规格（《2027 年足式机器人挑战赛 V1.0》表 1）原文**：

> 限高框架内高 290mm，内宽 800mm，长 800mm。限高框架颜色为绿色。

**通过判定（表 2）**：

> 机器人以任何运动姿态从限高框架下穿过（允许触碰，但不允许暴力撞击，
> 例如导致框架出现移位），则表示限高框架障碍通过完成。

| 项目 | 规格 |
|---|---|
| 内高 | **290 mm** |
| 内宽 | 800 mm |
| 内长 | 800 mm |
| 颜色 | 绿色 |
| 允许触碰 | 是（但不能撞到框架移位） |

**290mm 意味着什么**：机器人正常站立时总高约 30~35cm，必须**趴到 29cm 以下**才能过去。
这不是"跳"或"跨"的障碍，是纯粹考验**姿态控制**的障碍。

---

## 2. 和旧版（错的）的对比

这个场地之前建错了，2026-11-28 对照规则和官方模型 `Robocon2027.xml` 重写：

| 项目 | 旧版（错） | 官方规格（现在） |
|---|---|---|
| 结构 | 一根圆柱横杆 + 两个支座 | **800×800 方形框架**（4 柱 + 顶梁 + 中间加强梁） |
| 内高 | 0.25 → 0.35 m | **0.29 m**（官方） |
| 宽度 | 1.0 m | **0.80 m** |
| 颜色 | 紫色 | **绿色** |

---

## 3. 几何细节（官方模型实测）

```
4 根立柱   size=(0.02, 0.02, 0.145)，中心 z=0.145  → 柱高 0.29
4 根顶梁   z=0.31，半高 0.02                      → 梁底 = 0.29（正好是内高）
外沿尺寸   0.84 × 0.84
内空尺寸   0.80 × 0.80
另有       1 根中间横梁 + 2 根中间立柱（都在顶部，加强用，不挡通行）
```

---

## 4. 训练命令

任务名：**`Robot-16DOF-LowBar-v0`**

```powershell
$root="D:\Project\nano\16dof-rl"; $py="D:\Project\wheelleg-rl\.venv\Scripts\python.exe"
$env:PYTHONPATH="$root\src;$root\mjlab\src"; $env:PYTHONIOENCODING="utf-8"
$env:WARP_CACHE_PATH="$root\.cache\warp"; $env:TMP="D:\Project\nano\.tmp"; $env:TEMP=$env:TMP
Set-Location $root

& $py -u train_16dof.py Robot-16DOF-LowBar-v0 --agent.max-iterations 1200 --agent.run-name "lowbar_第一轮"
```

**难度课程**：难度 0 内高 0.36 m（好钻）→ 难度 1 内高 0.29 m（官方标准）。

---

## 5. 判据为什么只写"前进 + 姿态"，不加"蹲低奖励"

框架是真的碰撞体，站高了会被顶梁挡住、根本走不过去，前进奖励自然上不来。
**物理约束本身就是最好的老师**，硬加一个"趴低给分"反而可能教出"趴到刮地"的怪行为
（项目里那个 `crouch_height_reward` 绑定在"模式指令"系统上，这个场地没有那套指令，用不了）。

---

## 6. 第一次训练要盯什么

1. **可能卡在框架前面不敢进**。290mm 对它来说很矮，初期大概率在框前反复试探。
   看 `Curriculum/terrain_levels` 是不是卡在 0~2 不动。
2. **可能"刮过去"**。允许触碰，但如果它学会用身体蹭着梁硬挤过去，
   到了实场容易把框架撞移位（规则里算犯规，罚 20 分）。
3. **和梅花桩的"趴"是两回事**。梅花桩练的是低姿态落点，这里练的是低姿态前进，
   共同点是都要趴得下去，但动作模式不同，建议单独训练而不是接梅花桩的模型。

---

## 7. 怎么看这个场地

双击本文件夹里的 `查看场地.bat`，或者：

```powershell
cd D:\Project\nano\16dof-rl
$env:PYTHONPATH="D:\Project\nano\16dof-rl\src;D:\Project\nano\16dof-rl\mjlab\src"
python preview_terrain.py lowbar              # 看最难（官方 290mm）
python preview_terrain.py lowbar --level 0    # 看最简单（360mm）
python preview_terrain.py lowbar --levels 4   # 四个难度并排
```

窗口操作：左键旋转、右键平移、滚轮缩放、Esc 关闭。

---

## 8. 文件清单

```
限高框架/场地/
├── README.md                       ← 本文档
├── 查看场地.bat                    ← 双击打开 3D 预览
├── src/robot/terrains/
│   └── competition_terrains.py     ← 地形（RCLowBarTerrainCfg）
└── config/
    └── env_cfg_lowbar.py           ← 训练配置
```

**集成方法**：`competition_terrains.py` 复制到 `src/robot/terrains/`（覆盖同名文件，
注意这个文件里还有高墙 `RCWallTerrainCfg`，两个场地共用）；配置粘到 `env_cfgs.py` 的高墙段之后。