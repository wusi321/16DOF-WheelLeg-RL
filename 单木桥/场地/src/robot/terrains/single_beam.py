"""单木桥（single beam）地形：一条窄长的木头横在场地中间，机器人要走过去。

官方规格（《2027 年足式机器人挑战赛 V1.0》3.1 节障碍赛）原文：

    单木桥长 3000mm，宽 400mm，高 100mm。表面涂绿色水性漆。
    称 400mm 为短边，3000mm 为长边。

通过判定（规则 4.2 节）：
    机器人从一侧短边进入单木桥，并从另一端短边离开，则表示障碍通过完成。
    通过单木桥时，机器人从障碍物侧面摔下，或 1/2 以上足与地面接触（含 1/2），
    则需重新通过。

所以难点不是"跨过去"，而是**在 400mm 宽的窄面上保持平衡走完 3 米**。
两侧没有护栏，掉下去就是地面（与桥面差 100mm）；四条腿里两条着地就算失败。

训练意义上，这是所有场地里最考验**横向稳定和持续纠偏**的一个：
梅花桩允许"踩一下就走"，单木桥要求"持续走直线"。

难度设计说明：规则尺寸固定（400mm 宽不变），
所以课程靠**桥高 0.06 → 0.10 m** 和**桥宽 0.40 → 0.32 m**。
宽度训练时略窄于官方 400mm，等于给比赛留出余量。
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeometry, TerrainOutput
from mjlab.terrains.utils import make_plane
from mjlab.utils.color import brand_ramp

_BEAM_COLOR = (0.13, 0.55, 0.24)  # 绿色水性漆，和规则一致
_GROUND_COLOR = (0.85, 0.80, 0.30, 1.0)  # 场地黄色


@dataclass(kw_only=True)
class BoxSingleBeamTerrainCfg(SubTerrainCfg):
    """一条窄长木桥（官方 3000 × 400 × 100 mm），两端无斜坡，需要抬腿上桥。"""

    size: tuple[float, float] = (8.0, 1.6)
    """patch 尺寸 (x, y)，单位 m。"""

    start_length: float = 0.8
    """起点平地长度（桥之前）。"""

    beam_length: float = 3.0
    """桥长（m）。官方 3000mm。"""

    width_range: tuple[float, float] = (0.40, 0.32)
    """桥宽范围（m）：难度 0 → 0.40（官方），难度 1 → 0.32（更窄，练余量）。"""

    height_range: tuple[float, float] = (0.06, 0.10)
    """桥高范围（m）：难度 0 → 0.06（容易爬上去），难度 1 → 0.10（官方）。"""

    def function(
        self,
        difficulty: float,
        spec: mujoco.MjSpec,
        rng: np.random.Generator,
    ) -> TerrainOutput:
        del rng
        body = spec.body("terrain")
        geometries: list[TerrainGeometry] = []

        _sx, sy = self.size
        cy = sy / 2.0
        d = float(np.clip(difficulty, 0.0, 1.0))

        w0, w1 = self.width_range
        h0, h1 = self.height_range
        width = w0 + (w1 - w0) * d
        height = h0 + (h1 - h0) * d

        # 1) 场地地面（黄色，与官方场地一致）
        for box in make_plane(body, self.size, 0.0, center_zero=False):
            geometries.append(TerrainGeometry(geom=box, color=_GROUND_COLOR))

        # 2) 木桥本体：居中，沿 x 方向铺开，顶面高出地面 height
        x0 = self.start_length + 0.5
        beam_cx = x0 + self.beam_length / 2.0
        beam = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(self.beam_length / 2.0, width / 2.0, height / 2.0),
            pos=(beam_cx, cy, height / 2.0),
        )
        geometries.append(TerrainGeometry(geom=beam, color=brand_ramp(_BEAM_COLOR, d)))

        origin = np.array([self.start_length / 2.0, cy, 0.0])
        return TerrainOutput(origin=origin, geometries=geometries)