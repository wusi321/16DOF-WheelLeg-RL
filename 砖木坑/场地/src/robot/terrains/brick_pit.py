"""砖木坑（官方规则表 1，**L 形**）——2026-11-28 按官方规格新建。

**官方规格原文**（《2027 年足式机器人挑战赛 V1.0》表 1）：

    砖木坑 2000mm × 2000mm。L 形砖木坑的护栏内分两个区，两侧分别填有碎砖头、
    腐熟松树皮（10-20mm）。短边宽 1000mm，长边长 2000mm。护栏高度为 150mm，颜色为绿色。

**通过定义**（表 2）：

    机器人必须从砖木坑的**短边**进入或离开。
    跨越过程中，若机器人足数的 1/2 以上足端（含 1/2）与**坑外地面**接触，则越障失败。

**几何还原**（依据官方模型 ``Robocon2027.xml`` 的护栏坐标）：

    L 形轮廓，整体外接 2000 × 2000mm，每条臂宽 1000mm：
        竖臂  x ∈ [0.50, 1.50], y ∈ [0.30, 2.30]   （宽 1000 × 长 2000）
        横臂  x ∈ [1.50, 2.50], y ∈ [1.30, 2.30]   （长 1000 × 宽 1000）

    护栏 6 段，高 150mm、厚 25mm，把 L 形围成闭环（机器人要越过护栏进/出）：
        XML 实测：pit_wall_left / top / right_top / inner / right_bot / bot
        以及一道 45° 斜向分隔墙 pit_separator，把坑内分成"碎砖区"和"树皮区"。

**填充材料的处理方式**：

规则要求填"碎砖头、腐熟松树皮（10-20mm）"，这些都是**松散颗粒**，MuJoCo 里没法
真的模拟颗粒流。这里的做法是：在坑内铺一层随机大小的**小方块**（边长 30mm 左右，
高度 40~80mm 随机），并给低摩擦，近似"踩上去会陷、会滑"的感觉。

这只能是**近似**——真实颗粒的塌陷、侧向流动模拟不出来。如果训练时发现机器人
在坑里打滑到无法前进，优先调 `fill_friction` 而不是加奖励。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeometry, TerrainOutput
from mjlab.terrains.utils import make_plane

_GUARD_COLOR = (0.13, 0.55, 0.24, 1.0)    # 护栏绿色（规则要求）
_BRICK_COLOR = (0.62, 0.42, 0.36, 1.0)    # 碎砖头：偏红灰
_BARK_COLOR = (0.45, 0.33, 0.20, 1.0)     # 腐熟松树皮：深棕
_GROUND = (0.35, 0.35, 0.35, 1.0)


@dataclass(kw_only=True)
class BoxBrickPitTerrainCfg(SubTerrainCfg):
    """L 形砖木坑：护栏高 150mm，坑内两区分别铺碎砖/树皮碎块。"""

    size: tuple[float, float] = (4.0, 3.0)
    """patch 尺寸。L 形本体 2×2m，前方留出进入空间。"""

    arm_width: float = 1.00
    """每条臂的宽度（m）。官方 1000mm。"""

    arm_length: float = 2.00
    """长边长度（m）。官方 2000mm。"""

    guard_height: float = 0.15
    """护栏高度（m）。官方 150mm。"""

    guard_thickness: float = 0.025
    """护栏厚度（m）。XML 实测 25mm。"""

    fill_height_range: tuple[float, float] = (0.04, 0.08)
    """填充碎块的高度范围（m）。规则说材料粒径 10-20mm，但太薄踩不出"松散"感，
    这里放大到 40~80mm 才能让轮子感到起伏。"""

    fill_block: float = 0.045
    """单个碎块的边长（m）。太大看不出松散，太小几何体数量爆炸（每个 patch 的
    几何体数会被 num_rows 放大：10 档课程 × 700 个 = 7000 个 geom，会明显拖慢训练）。"""

    fill_friction: float = 0.45
    """填充材料的摩擦系数。比硬地面（0.8）低，模拟打滑。"""

    fill_density: float = 0.55
    """填充密度（0~1）。1.0 = 铺满，越小越稀疏、坑越"空"。"""

    _counter: int = field(default=0, init=False, repr=False)

    def _layout(self, difficulty: float) -> dict:
        """L 形布局。难度只影响填充密度（材料越多越松），尺寸是官方固定值。"""
        d = float(np.clip(difficulty, 0.0, 1.0))
        sx, sy = self.size
        aw = self.arm_width
        al = self.arm_length

        # 竖臂在左下，横臂在右上，拐角共享
        x0 = 0.50                     # 竖臂左边界
        y0 = 0.30                     # 竖臂下边界
        vx = (x0, x0 + aw)            # 竖臂 x 范围
        vy = (y0, y0 + al)            # 竖臂 y 范围
        hx = (x0 + aw, x0 + aw + (al - aw))   # 横臂 x 范围（长 1000）
        hy = (y0 + (al - aw), y0 + al)        # 横臂 y 范围（宽 1000）

        return {
            "d": d,
            "vertical": (vx, vy),
            "horizontal": (hx, hy),
            # 分隔墙：从 (x0, vy[1]) 到 (vx[1], hy[0])，即左上角到拐角内侧的 45° 斜线
            "sep_line": (x0, vy[1], vx[1], hy[0]),
        }

    def _in_pit(self, x: float, y: float, lay: dict) -> bool:
        (vx, vy), (hx, hy) = lay["vertical"], lay["horizontal"]
        in_v = vx[0] <= x <= vx[1] and vy[0] <= y <= vy[1]
        in_h = hx[0] <= x <= hx[1] and hy[0] <= y <= hy[1]
        return in_v or in_h

    def _above_separator(self, x: float, y: float, lay: dict) -> bool:
        """判断点是否在 45° 分隔墙的"左上"一侧（碎砖区）。"""
        x1, y1, x2, y2 = lay["sep_line"]
        # 线方向 (x2-x1, y2-y1)，取法向判断侧别
        return (y - y1) * (x2 - x1) - (x - x1) * (y2 - y1) > 0

    def function(
        self,
        difficulty: float,
        spec: mujoco.MjSpec,
        rng: np.random.Generator,
    ) -> TerrainOutput:
        body = spec.body("terrain")
        geometries: list[TerrainGeometry] = []
        lay = self._layout(difficulty)
        d = lay["d"]
        (vx, vy), (hx, hy) = lay["vertical"], lay["horizontal"]
        t = self.guard_thickness
        gh = self.guard_height

        # 1) 地面
        for box in make_plane(body, self.size, 0.0, center_zero=False):
            geometries.append(TerrainGeometry(geom=box, color=_GROUND))

        def add_guard(x0: float, x1: float, y0: float, y1: float) -> None:
            """加一段护栏（按两端点给定，厚度自动取 guard_thickness）。"""
            cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
            hxs = max(t / 2, abs(x1 - x0) / 2)
            hys = max(t / 2, abs(y1 - y0) / 2)
            g = body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=(hxs, hys, gh / 2),
                pos=(cx, cy, gh / 2),
            )
            geometries.append(TerrainGeometry(geom=g, color=_GUARD_COLOR))

        # 2) 六段护栏（严格按 XML 的六段墙，围成 L 形闭环）
        add_guard(vx[0], vx[0], vy[0], vy[1])                 # 左
        add_guard(hx[0], hx[1], vy[1], vy[1])                 # 上（横臂顶）
        add_guard(hx[1], hx[1], hy[0], hy[1])                 # 右（横臂端，短边之一）
        add_guard(vx[1], hx[1], hy[0], hy[0])                 # 内（横臂底）
        add_guard(vx[1], vx[1], vy[0], hy[0])                 # 右（竖臂段）
        add_guard(vx[0], vx[1], vy[0], vy[0])                 # 下（竖臂底，短边之二）

        # 3) 45° 分隔墙
        x1, y1, x2, y2 = lay["sep_line"]
        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        half_len = 0.5 * float(np.hypot(x2 - x1, y2 - y1))
        sep = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(half_len, t / 2, gh / 2),
            pos=(cx, cy, gh / 2),
            euler=(0.0, 0.0, float(np.arctan2(y2 - y1, x2 - x1))),
        )
        geometries.append(TerrainGeometry(geom=sep, color=_GUARD_COLOR))

        # 4) 坑内填充碎块：网格撒点 + 抖动，按分隔墙分两种材料
        step = self.fill_block * 2.0
        n_side = int(np.ceil(self.arm_length / step))
        density = self.fill_density * (1.0 - 0.25 * d)   # 越难越稀疏（更松散）
        count = 0
        for ix in range(-n_side, n_side + 1):
            for iy in range(-n_side, n_side + 1):
                px = vx[0] + 0.5 * step + ix * step + rng.uniform(-0.008, 0.008)
                py = vy[0] + 0.5 * step + iy * step + rng.uniform(-0.008, 0.008)
                if not self._in_pit(px, py, lay):
                    continue
                if rng.random() > density:
                    continue
                fh = float(rng.uniform(*self.fill_height_range))
                color = _BRICK_COLOR if self._above_separator(px, py, lay) else _BARK_COLOR
                g = body.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=(self.fill_block / 2, self.fill_block / 2, fh / 2),
                    pos=(px, py, fh / 2),
                )
                g.friction = (self.fill_friction, 0.05, 0.05)
                geometries.append(TerrainGeometry(geom=g, color=color))
                count += 1

        # 起点：横臂 +x 端（短边）外侧
        origin = np.array([hx[1] + 0.55, 0.5 * (hy[0] + hy[1]), 0.0])
        return TerrainOutput(origin=origin, geometries=geometries)