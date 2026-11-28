"""沟壑组合（官方规则表 1，**L 形**）——2026-11-28 按官方规格重写。

**官方规格原文**（《2027 年足式机器人挑战赛 V1.0》表 1）：

    沟壑组合障碍呈 L 形，两顶部平台高度均为 360mm，长宽均为 800mm。
    斜坡侧平台为启动区 2，刷有黄色水性漆，外围是红色虚线。
    台阶深度 320mm、高度 120mm，宽度 800mm。
    沟壑为两顶部平台之间的间隔，宽度为 360mm。斜坡坡度为 20 度。
    除启动区 2 外，表面均刷绿色水性漆。

**通过定义**（表 2）：

    机器人可从任一方向迈上台阶/斜坡，通过沟壑后，从另一方向的斜坡/台阶回到地面。
    机器人在通过台阶时，要求至少每一级台阶顶面与机器人足底有一次接触。
    跨越过程中，若机器人足数的 1/2 以上足与地面接触（含 1/2），则越障失败。

**几何还原**（依据官方模型 ``Robocon2027.xml`` 的精确坐标）：

    沿 -x 方向：台阶(顶0.12) → 台阶(顶0.24) → 平台A(顶0.36) → 沟(宽0.36) → 平台B(顶0.36)
    然后从平台 B 沿 **+y** 方向接一段 20° 斜坡下到地面
    —— 台阶/平台/沟这一串在 x 方向，斜坡拐到 y 方向，这就是"L 形"的由来。

XML 实测数据：
    stairs_07_00  size=(0.16, 0.40, 0.06)  pos=(-4.16, -2, 0.06)   → 深0.32 宽0.80 顶0.12
    stairs_07_01  size=(0.16, 0.40, 0.12)  pos=(-4.48, -2, 0.12)   → 深0.32 宽0.80 顶0.24
    platform_08   size=(0.40, 0.40, 0.18)  pos=(-5.04, -2, 0.18)   → 0.80x0.80 顶0.36
    platform_09   size=(0.40, 0.40, 0.18)  pos=(-6.20, -2, 0.18)   → 0.80x0.80 顶0.36
    两平台间隙 = 0.36 ✓ 与规则的"沟壑宽 360mm"完全吻合
    ramp_10       20° 斜坡，水平投影 0.989m、宽 0.80、高 0.36，接在平台 B 的 +y 侧
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeometry, TerrainOutput
from mjlab.terrains.utils import make_plane
from mjlab.utils.color import brand_ramp

_GREEN = (0.13, 0.55, 0.24, 1.0)      # 绿色水性漆
_YELLOW = (0.90, 0.80, 0.15, 1.0)     # 启动区 2（黄色）
_GROUND = (0.35, 0.35, 0.35, 1.0)
_GAP_BOTTOM = (0.12, 0.12, 0.12, 1.0)  # 沟底暗色，方便看出"这是坑"


@dataclass(kw_only=True)
class BoxGapCourseTerrainCfg(SubTerrainCfg):
    """L 形沟壑组合：台阶 2 级 + 平台 ×2（顶 360mm）+ 沟 360mm + 20° 斜坡。"""

    size: tuple[float, float] = (5.2, 3.2)
    """patch 尺寸 (x, y)。要放 3.9m 的台阶-平台-沟序列，再往 +y 伸出 0.99m 的斜坡。"""

    # ---- 官方固定尺寸（难度只缩放高度/沟宽，尺寸比例保持不变）----
    platform_height: float = 0.36
    """平台高度（m）。官方 360mm。"""

    platform_size: float = 0.80
    """平台长宽（m）。官方 800×800mm。"""

    gap_width: float = 0.36
    """两平台之间的沟壑宽度（m）。官方 360mm。"""

    stair_depth: float = 0.32
    """台阶深度（m）。官方 320mm。"""

    stair_height: float = 0.12
    """台阶高度（m）。官方 120mm。"""

    ramp_angle_deg: float = 20.0
    """斜坡坡度（度）。官方 20°。"""

    height_scale_min: float = 0.50
    """难度 0 时把高度/沟宽缩到官方的多少倍。0.5 → 平台 180mm、沟 180mm。"""

    _counter: int = field(default=0, init=False, repr=False)

    def _scale(self, difficulty: float) -> float:
        d = float(np.clip(difficulty, 0.0, 1.0))
        return self.height_scale_min + (1.0 - self.height_scale_min) * d

    def _layout(self, difficulty: float) -> dict:
        s = self._scale(difficulty)
        h = self.platform_height * s          # 平台高度
        gap = self.gap_width * s              # 沟宽
        sh = self.stair_height * s            # 台阶高
        sd = self.stair_depth                 # 台阶深（官方固定，不缩放）
        ps = self.platform_size               # 平台边长（官方固定）
        sx, sy = self.size

        cy = sy * 0.32                        # 平台中心 y（下方留白，上方给斜坡）
        # 从 +x 往 -x 排：台阶2 → 台阶1 → 平台A → 沟 → 平台B
        x_stair1_r = sx - 0.55                # 最外侧台阶的右边界
        x_stair1_l = x_stair1_r - sd
        x_stair2_l = x_stair1_l - sd
        x_platA_r = x_stair2_l
        x_platA_l = x_platA_r - ps
        x_platB_r = x_platA_l - gap
        x_platB_l = x_platB_r - ps

        return {
            "h": h, "gap_w": gap, "sh": sh, "sd": sd, "ps": ps, "cy": cy,
            "stair1": (x_stair1_l, x_stair1_r),
            "stair2": (x_stair2_l, x_stair1_l),
            "platA": (x_platA_l, x_platA_r),
            "platB": (x_platB_l, x_platB_r),
            "gap_x": (x_platB_r, x_platA_l),
        }

    def function(
        self,
        difficulty: float,
        spec: mujoco.MjSpec,
        rng: np.random.Generator,
    ) -> TerrainOutput:
        del rng
        body = spec.body("terrain")
        geometries: list[TerrainGeometry] = []
        lay = self._layout(difficulty)
        h = lay["h"]
        gap_w = lay["gap_w"]
        sh = lay["sh"]
        ps = lay["ps"]
        cy = lay["cy"]
        sx, sy = self.size
        green = _GREEN

        # 1) 地面
        for box in make_plane(body, self.size, 0.0, center_zero=False):
            geometries.append(TerrainGeometry(geom=box, color=_GROUND))

        def add_box(x0: float, x1: float, y0: float, y1: float, top: float, color):
            """按"顶面高度"加一个实心方块（底到底面 0）。"""
            cx, cyy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
            hx, hy = 0.5 * (x1 - x0), 0.5 * (y1 - y0)
            g = body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=(hx, hy, 0.5 * top),
                pos=(cx, cyy, 0.5 * top),
            )
            geometries.append(TerrainGeometry(geom=g, color=color))
            return g

        # 2) 两级台阶（官方：深 320、高 120，宽 800）
        s1 = lay["stair1"]
        s2 = lay["stair2"]
        add_box(s1[0], s1[1], cy - ps / 2, cy + ps / 2, sh, green)
        add_box(s2[0], s2[1], cy - ps / 2, cy + ps / 2, 2 * sh, green)

        # 3) 平台 A（斜坡对侧）
        pa = lay["platA"]
        add_box(pa[0], pa[1], cy - ps / 2, cy + ps / 2, h, green)

        # 4) 平台 B（启动区 2，黄色 + 红色虚线——这里用黄色顶面表示）
        pb = lay["platB"]
        add_box(pb[0], pb[1], cy - ps / 2, cy + ps / 2, h, green)
        # 启动区 2 的黄色标记面（薄片，放在平台 B 顶面上）
        ymark = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(ps / 2 - 0.02, ps / 2 - 0.02, 0.002),
            pos=(0.5 * (pb[0] + pb[1]), cy, h + 0.002),
            contype=0,
            conaffinity=0,
        )
        geometries.append(TerrainGeometry(geom=ymark, color=_YELLOW))

        # 5) 沟壑底部示意（两平台之间挖空到地面，所以这里什么都不加，
        #    只在地面那一层已经存在；加一条暗色薄片方便视觉上看出沟）
        gx0, gx1 = lay["gap_x"]
        gmark = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(0.5 * (gx1 - gx0), ps / 2, 0.0015),
            pos=(0.5 * (gx0 + gx1), cy, 0.0015),
            contype=0,
            conaffinity=0,
        )
        geometries.append(TerrainGeometry(geom=gmark, color=_GAP_BOTTOM))

        # 6) 20° 斜坡：从平台 B 的 +y 边缘向外下到地面
        ang = math.radians(self.ramp_angle_deg)
        run = h / math.tan(ang)              # 水平投影长度
        slope_len = h / math.sin(ang)        # 斜面长度
        y_edge = cy + ps / 2                 # 平台 B 的上边缘
        ramp_cx = 0.5 * (pb[0] + pb[1])
        ramp_cy = y_edge + run / 2
        ramp_cz = h / 2
        rg = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(ps / 2, slope_len / 2, 0.03),
            pos=(ramp_cx, ramp_cy, ramp_cz),
            euler=(-ang, 0.0, 0.0),
        )
        geometries.append(TerrainGeometry(geom=rg, color=green))

        # 起点：台阶外侧，机器人从这里出发沿 -x 走，跨沟、下斜坡
        origin = np.array([sx - 0.25, cy, 0.0])
        return TerrainOutput(origin=origin, geometries=geometries)