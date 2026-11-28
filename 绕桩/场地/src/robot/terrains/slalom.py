"""绕桩（slalom）地形：L 形排列的 6 根竖桩 + 3 个圆形必达区。

**规格全部来自官方规则，不是估的**（《2027 年足式机器人挑战赛 V1.0》表 1）：

    竖桩布局为 L 形。每个竖桩的直径为 100mm、高度为 100mm，竖桩之间的中心距 900mm。
    绕桩区的两端及拐角外侧各有一个圆形必达区（红色虚线圆），直径为 300mm。
    竖桩颜色为绿色。

官方场地模型 ``Robocon2027.xml`` 里的精确坐标（世界坐标，已换算成局部布局）：

    拐角桩  (6.0, -2.0)
    水平向 -X 三根：(5.1, -2.0) (4.2, -2.0) (3.3, -2.0)   → 中心距 0.9
    竖直向 +Y 两根：(6.0, -1.1) (6.0, -0.2)               → 中心距 0.9
    必达区 A 水平端外侧 (2.85, -2.0)
    必达区 B 竖直端外侧 (6.0, 0.25)
    必达区 C 拐角外侧   (6.3182, -2.3182)     ← 沿对角外偏 0.45

机器人从水平端外侧（A 附近）进入，绕过所有桩走到竖直端外侧（B）。
三个必达区都必须踩到，**顺序是 A → C → B**（先到拐角，再到竖直端）。

实现上有个关键约定（判据 `mdp/slalom.py` 按几何签名识别，改了就对不上）：

* **必达区标记**：不参与碰撞、极薄、截面半宽正好 ``WAYPOINT_HALF = 0.02`` 的小方块。
  判据靠 ``size[0] == 0.02 且 size[2] <= 0.01 且不碰撞`` 认它；
  视觉上的红色圆盘是「不碰撞 + 半径 0.15」，尺寸不同，不会被误认成标记，也不会被认成柱子。
* **竖桩**：参与碰撞、半径 <= 0.06、半高 >= 0.05 的圆柱。半径 0.05 / 半高 0.05 正好符合。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeometry, TerrainOutput
from mjlab.terrains.utils import make_plane
from mjlab.utils.color import brand_ramp

#: 必达区标记的截面半宽。必须和 mdp/slalom.py 的 WAYPOINT_HALF 完全一致。
WAYPOINT_HALF = 0.02

_PILLAR_COLOR = (0.13, 0.55, 0.24)  # 竖桩绿色（规则要求）
_GROUND_COLOR = (0.35, 0.35, 0.35, 1.0)
_REACH_COLOR = (0.90, 0.10, 0.10, 1.0)  # 必达区红色


@dataclass(kw_only=True)
class BoxSlalomTerrainCfg(SubTerrainCfg):
    """L 形绕桩：6 根竖桩（直径 100mm、高 100mm、中心距 900mm）+ 3 个必达区（直径 300mm）。"""

    size: tuple[float, float] = (6.0, 3.0)
    """patch 尺寸 (x, y)。L 形要放 2.7m 长的水平臂和 1.8m 高的竖直臂，所以比别的场地宽。"""

    pillar_radius: float = 0.05
    """竖桩半径（m）。官方直径 100mm。"""

    pillar_height: float = 0.10
    """竖桩高度（m）。官方 100mm。注意只有 10cm，轮子能压过去——难点在"绕过"而不是"跨越"。"""

    pillar_pitch: float = 0.90
    """竖桩中心距（m）。官方 900mm，固定值。"""

    reach_radius: float = 0.15
    """必达区半径（m）。官方直径 300mm。"""

    x_arm: int = 4
    """水平臂上的桩数（含拐角）。官方 4 根。"""

    y_arm: int = 2
    """竖直臂上**额外**的桩数（不含拐角）。官方 2 根。"""

    #: 生成计数，用于给 geom 起唯一名字（便于调试时按名字找）。
    _counter: int = field(default=0, init=False, repr=False)

    def _layout(self, difficulty: float) -> dict:
        """把 L 形布局算成局部坐标。

        难度只影响必达区半径（判定宽严），桩的尺寸/间距是官方固定值，不做缩放——
        绕桩的难点本来就不靠尺寸变化，靠的是"能不能走 S 形不碰桩"。
        """
        d = float(np.clip(difficulty, 0.0, 1.0))
        sx, sy = self.size
        # 拐角桩位置：给左侧留出进入空间、上方留出竖直臂空间
        corner_x = sx * 0.68
        corner_y = sy * 0.34
        pitch = self.pillar_pitch

        pillars = [(corner_x, corner_y)]
        for i in range(1, self.x_arm):
            pillars.append((corner_x - i * pitch, corner_y))
        for i in range(1, self.y_arm + 1):
            pillars.append((corner_x, corner_y + i * pitch))

        # 必达区：水平端外侧、拐角外侧（沿对角外偏 0.45）、竖直端外侧
        out_h = 0.45
        last_x = corner_x - (self.x_arm - 1) * pitch
        last_y = corner_y + self.y_arm * pitch
        reaches = [
            (last_x - out_h, corner_y),                       # A 水平端外侧
            (corner_x + out_h * 0.7071, corner_y - out_h * 0.7071),  # C 拐角外侧
            (corner_x, last_y + out_h),                       # B 竖直端外侧
        ]
        return {"pillars": pillars, "reaches": reaches, "d": d}

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
        d = lay["d"]

        # 1) 地面
        for box in make_plane(body, self.size, 0.0, center_zero=False):
            geometries.append(TerrainGeometry(geom=box, color=_GROUND_COLOR))

        # 2) 竖桩（圆柱，参与碰撞）
        pillar_color = brand_ramp(_PILLAR_COLOR, d)
        for x, y in lay["pillars"]:
            g = body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=(self.pillar_radius, self.pillar_height / 2.0),
                pos=(x, y, self.pillar_height / 2.0),
            )
            geometries.append(TerrainGeometry(geom=g, color=pillar_color))

        # 3) 必达区：每个放两样东西
        #    (a) 不碰撞的红色薄圆盘 —— 视觉标记，人看着知道该踩哪
        #    (b) 不碰撞的极小薄方块（截面 0.02）—— 判据靠它识别必经点
        sx, sy = self.size
        for x, y in lay["reaches"]:
            disk = body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=(self.reach_radius, 0.0015),
                pos=(x, y, 0.0015),
                contype=0,
                conaffinity=0,
            )
            geometries.append(TerrainGeometry(geom=disk, color=_REACH_COLOR))

            mark = body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=(WAYPOINT_HALF, WAYPOINT_HALF, 0.002),
                pos=(x, y, 0.004),
                contype=0,
                conaffinity=0,
            )
            geometries.append(TerrainGeometry(geom=mark, color=_REACH_COLOR))

        # 起点：水平臂外侧留出的进入区
        origin = np.array([0.35, lay["pillars"][0][1], 0.0])
        return TerrainOutput(origin=origin, geometries=geometries)