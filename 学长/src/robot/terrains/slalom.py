"""绕桩课程（slalom）：一排方柱，机器人要走 S 形依次绕过每根柱子。

规则对齐"足式障碍赛"：
- 柱子按赛道宽度要求做成方形柱（默认 50 × 50 mm、高 300 mm，同横杆规格）；
- 每根柱子旁边有一个**必经点**（waypoint），左右交替，必须**按顺序**经过；
- 碰到柱子扣分；跳点（越过下一个必经点、直接吃到更后面的点）重罚；
- 走的必经点越多，累计得分越高；难度提升靠**缩短柱距**。

地形里会额外放一批**不参与碰撞的很小方块**作为必经点标记，MDP 侧
（`src/robot/mdp/slalom.py`）按几何签名把它们读出来，因此训练代码不需要
知道难度是多少，也不需要和地形参数对齐。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeometry, TerrainOutput
from mjlab.utils.color import brand_ramp

#: 难度 -> 柱距（x 方向，中心到中心）
_SPACING = np.array([1.00, 0.90, 0.80, 0.70, 0.60, 0.52])
_STAGE = np.array([0.00, 0.20, 0.40, 0.60, 0.80, 1.00])

_PILLAR_COLOR = (0.95, 0.45, 0.10)
_GROUND_COLOR = (0.35, 0.35, 0.35, 1.0)
_WAYPOINT_COLOR = (0.20, 0.55, 0.95, 1.0)

#: 必经点标记的尺寸（半长），MDP 侧按这个签名识别
WAYPOINT_HALF = 0.02
WAYPOINT_HEIGHT = 0.002


@dataclass(kw_only=True)
class BoxSlalomTerrainCfg(SubTerrainCfg):
    """一排方柱 + 左右交替的必经点标记。"""

    size: tuple[float, float] = (8.0, 1.6)

    start_length: float = 0.8
    end_length: float = 0.8

    pillar_size: tuple[float, float] = (0.05, 0.05)
    """柱子截面（x, y），默认 50 × 50 mm（足式障碍赛横杆规格）。"""

    pillar_height: float = 0.30
    """柱子高度；高于轮半径即可，碰到就说明走歪了。"""

    spacing_range: tuple[float, float] = (1.00, 0.52)
    """柱距范围（仅文档/校验，实际曲线见 `_SPACING`）。"""

    waypoint_offset: float = 0.42
    """必经点相对中心线的横向偏移（左右交替）。"""

    ground_height: float = 0.0
    """地面高度（绕桩是平地，不挖坑）。"""

    _pillar_counter: int = field(default=0, init=False, repr=False)
    _waypoint_counter: int = field(default=0, init=False, repr=False)

    def spacing(self, difficulty: float) -> float:
        d = float(np.clip(difficulty, 0.0, 1.0))
        return float(np.interp(d, _STAGE, _SPACING))

    def function(
        self,
        difficulty: float,
        spec: mujoco.MjSpec,
        rng: np.random.Generator,
    ) -> TerrainOutput:
        body = spec.body("terrain")
        geometries: list[TerrainGeometry] = []

        sx, sy = self.size
        cy = sy / 2.0
        top = float(self.ground_height)
        spacing = self.spacing(difficulty)
        phx, phy = self.pillar_size[0] / 2.0, self.pillar_size[1] / 2.0

        # 地面
        ground = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(sx / 2.0, sy / 2.0, 0.05),
            pos=(sx / 2.0, cy, top - 0.05),
        )
        geometries.append(TerrainGeometry(geom=ground, color=_GROUND_COLOR))

        x0 = self.start_length + 0.35
        x1 = sx - self.end_length - 0.35
        n = max(1, int(np.floor((x1 - x0) / spacing)) + 1)
        xs = np.linspace(x0, x1, n)
        pillar_half_z = self.pillar_height / 2.0

        for i, x in enumerate(xs):
            pillar = body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=(phx, phy, pillar_half_z),
                pos=(float(x), cy, top + pillar_half_z),
            )
            pillar.name = f"slalom_pillar_{self._pillar_counter}"
            self._pillar_counter += 1
            geometries.append(
                TerrainGeometry(
                    geom=pillar,
                    color=brand_ramp(_PILLAR_COLOR, 0.45 + 0.5 * float(np.clip(difficulty, 0.0, 1.0))),
                )
            )

            # 必经点：左右交替，贴在柱子旁边（顺序即通过顺序）
            side = 1.0 if (i % 2 == 0) else -1.0
            wy = cy + side * self.waypoint_offset
            wp = body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=(WAYPOINT_HALF, WAYPOINT_HALF, WAYPOINT_HEIGHT),
                pos=(float(x), wy, top + 0.02),
            )
            wp.contype = 0
            wp.conaffinity = 0
            wp.name = f"slalom_waypoint_{self._waypoint_counter}"
            self._waypoint_counter += 1
            geometries.append(TerrainGeometry(geom=wp, color=_WAYPOINT_COLOR))

        origin = np.array([self.start_length / 2.0, cy, top])
        return TerrainOutput(origin=origin, geometries=geometries)
