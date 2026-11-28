"""沟壑课程（gap course）：一块块平台 + 越来越宽的沟，车辆必须"越过去"。

与梅花桩的区别：平台是一条连续的带子（不用踩点），难点在**沟宽**——
从 5 cm 起步，最高 45 cm（接近 0.55 m 车长的 82%，必须靠抬腿/跃起跨过）。
掉进沟里会触发 `base_too_low` / `base_ground_contact` 终止（沟深 0.6 m）。

难度曲线（`difficulty` ∈ [0, 1]）：

    d = 0.00  沟宽 0.05 m（滚过去）
    d = 0.25  沟宽 0.18 m
    d = 0.50  沟宽 0.28 m
    d = 0.75  沟宽 0.38 m
    d = 1.00  沟宽 0.45 m（实场上限）

平台长度固定 0.40 m（"40 宽度"），宽度 = patch 宽度（默认 1.6 m）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeometry, TerrainOutput
from mjlab.utils.color import brand_ramp

#: 难度 -> 沟宽（线性插值控制点）
_GAP = np.array([0.05, 0.18, 0.28, 0.38, 0.45])
_STAGE = np.array([0.00, 0.25, 0.50, 0.75, 1.00])

_SLAB_COLOR = (0.13, 0.55, 0.24)  # 与赛事"绿色水性漆"一致
_FLOOR_COLOR = (0.18, 0.18, 0.20, 1.0)
_GOAL_COLOR = (0.95, 0.70, 0.15, 1.0)


@dataclass(kw_only=True)
class BoxGapCourseTerrainCfg(SubTerrainCfg):
    """平台 + 沟壑课程。"""

    size: tuple[float, float] = (8.0, 1.6)
    """patch 尺寸 (x, y)，单位 m。"""

    start_length: float = 0.8
    """起点平台长度。"""

    end_length: float = 0.8
    """终点平台长度。"""

    slab_length: float = 0.4
    """每块平台的长度（x 方向）。"""

    gap_range: tuple[float, float] = (0.05, 0.45)
    """沟宽范围（仅文档/校验，实际曲线见 `_GAP`）。"""

    pit_depth: float = 0.6
    """沟底相对平台顶面的下沉深度；掉下去即判失败。"""

    slab_height: float = 0.0
    """平台顶面高度（相对 patch origin 的地面）。0 = 与地面齐平。"""

    _slab_counter: int = field(default=0, init=False, repr=False)

    def gap_width(self, difficulty: float) -> float:
        d = float(np.clip(difficulty, 0.0, 1.0))
        return float(np.interp(d, _STAGE, _GAP))

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
        gap = self.gap_width(difficulty)
        top = float(self.slab_height)
        slab = float(self.slab_length)

        # 沟底（一整块），深度 pit_depth
        floor_z = top - self.pit_depth
        floor = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(sx / 2.0, sy / 2.0, 0.05),
            pos=(sx / 2.0, cy, floor_z - 0.05),
        )
        geometries.append(TerrainGeometry(geom=floor, color=_FLOOR_COLOR))

        thick = max(self.pit_depth, 0.05)
        half_z = thick / 2.0
        center_z = top - half_z

        def _slab(x0: float, length: float) -> None:
            if length <= 1.0e-3:
                return
            g = body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=(length / 2.0, sy / 2.0, half_z),
                pos=(x0 + length / 2.0, cy, center_z),
            )
            geometries.append(
                TerrainGeometry(
                    geom=g,
                    color=brand_ramp(_SLAB_COLOR, 0.35 + 0.5 * float(np.clip(difficulty, 0.0, 1.0))),
                )
            )

        # 起点 / 终点平台
        _slab(0.0, self.start_length)
        _slab(sx - self.end_length, self.end_length)

        # 中间：40 cm 平台 + gap 沟，交替铺满
        x = self.start_length
        limit = sx - self.end_length
        period = slab + gap
        while x + slab <= limit + 1.0e-6:
            _slab(x, min(slab, limit - x))
            x += period

        # 终点标记（纯视觉）
        goal_marker = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(0.06, sy / 2.0 - 0.05, 0.002),
            pos=(sx - self.end_length + 0.06, cy, top + 0.002),
        )
        goal_marker.contype = 0
        goal_marker.conaffinity = 0
        geometries.append(TerrainGeometry(geom=goal_marker, color=_GOAL_COLOR))

        origin = np.array([self.start_length / 2.0, cy, top])
        return TerrainOutput(origin=origin, geometries=geometries)
