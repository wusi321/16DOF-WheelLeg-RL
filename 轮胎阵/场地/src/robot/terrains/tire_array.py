"""轮胎阵（tire array）地形：平放的橡胶轮胎，机器人踩在环形胎面上通过。

规格**直接取自赛事官方场地模型** ``Robocon2027.xml``（不是估的）：

    单个轮胎
        外径 0.56 m（外半径 0.28，取自 mesh ``tire_ring_05_mesh`` 顶点）
        内孔直径 0.34 m（内半径 0.17，同上）
        整体高度 0.165 m

    官方碰撞体做法（``tire_col`` 类）
        不用环形网格做碰撞（物理引擎算环不划算），而是用 **12 个小圆柱** 围成圆环：
        圆柱半径 0.055、半高 0.0825 → 高 0.165，
        围成的环**中心半径 0.235**（所以外缘 0.29、内缘 0.18，与视觉模型对得上）

    阵列布局
        4 × 2 共 8 个轮胎，横向和纵向中心距都是 0.56 m（轮胎外径 = 间距，即彼此相切）
        官方坐标 X: -2.84 / -2.28 / -1.72 / -1.16，Y: -2.28 / -1.72

难点在这里：**胎面是窄环，宽度只有 (0.56-0.34)/2 = 0.11 m**，轮子必须踩在这个环上；
环中间是孔（深 0.165 m），踩空就陷进去。这跟梅花桩"踩方块"、跨沟"跳过去"都不一样。

课程难度靠**轮胎高度**从 0.06 m 长到 0.165 m：
轮胎低（6 cm）时轮子（直径 10.8 cm）能直接滚上去，等于平地；
轮胎高到 16.5 cm 就必须抬腿踩上窄胎面——这正是实场难度。
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeometry, TerrainOutput
from mjlab.terrains.utils import make_plane
from mjlab.utils.color import brand_ramp

#: 难度 -> 轮胎高度（米）。实场固定 0.165，训练从 0.06 起步让策略先学会"踩上去"。
_TIRE_STAGE = np.array([0.00, 0.25, 0.50, 0.75, 1.00])
_TIRE_HEIGHT = np.array([0.06, 0.09, 0.12, 0.15, 0.165])

#: 轮胎颜色（橡胶黑），刻意压暗，避免在训练可视化里刺眼。
_TIRE_COLOR = (0.10, 0.10, 0.12, 1.0)
_GROUND_COLOR = (0.35, 0.35, 0.35, 1.0)


@dataclass(kw_only=True)
class BoxTireArrayTerrainCfg(SubTerrainCfg):
    """4 × 2 轮胎阵列（轮胎平放，踩环形胎面通过）。"""

    size: tuple[float, float] = (8.0, 1.6)
    """patch 尺寸 (x, y)，单位 m。"""

    start_length: float = 0.8
    """起点平台长度（轮胎区之前）。"""

    num_x: int = 4
    """沿 x 方向的轮胎个数（实场 4 个）。"""

    num_y: int = 2
    """沿 y 方向的轮胎个数（实场 2 个，正好覆盖左右轮距 0.385）。"""

    pitch: float = 0.56
    """轮胎中心距（x、y 相同）。实场 = 轮胎外径，即彼此相切。"""

    tire_outer_r: float = 0.275
    """轮胎外半径（m）。**取规则值**：官方规则表 1 写"外径约 550mm"。

    注：官方模型 Robocon2027.xml 的碰撞体 ``tire_col`` 是 12 个半径 0.055 的圆柱
    摆在半径 0.235 的环上，外缘 0.235+0.055=0.29（直径 580mm），比规则值大约 5%。
    比赛场地按规则尺寸制作（规则第 7 条允许 ±5% 误差），所以训练用 0.275 更贴近实物。
    """

    tire_inner_r: float = 0.165
    """轮胎内孔半径（m）。官方：内径约 330mm（XML 碰撞体算出来是 0.18，同样偏大 5%）。"""

    height_range: tuple[float, float] = (0.06, 0.165)
    """轮胎高度范围（m），随难度线性增长。"""

    ring_balls: int = 12
    """每个轮胎用几个小圆柱拼成环（官方做法是 12 个）。"""

    @property
    def ring_radius(self) -> float:
        """小圆柱中心所在的环半径 = (外半径 + 内半径) / 2。"""
        return 0.5 * (self.tire_outer_r + self.tire_inner_r)

    @property
    def ball_radius(self) -> float:
        """小圆柱半径 = (外半径 - 内半径) / 2，保证外缘/内缘正好贴合轮胎轮廓。"""
        return 0.5 * (self.tire_outer_r - self.tire_inner_r)

    def function(
        self,
        difficulty: float,
        spec: mujoco.MjSpec,
        rng: np.random.Generator,
    ) -> TerrainOutput:
        del rng
        body = spec.body("terrain")
        geometries: list[TerrainGeometry] = []

        sx, sy = self.size
        cy = sy / 2.0
        d = float(np.clip(difficulty, 0.0, 1.0))
        h = float(np.interp(d, _TIRE_STAGE, _TIRE_HEIGHT))
        # 用 height_range 覆盖默认曲线（便于命令行/配置调参）
        h = self.height_range[0] + (self.height_range[1] - self.height_range[0]) * (
            (h - _TIRE_HEIGHT[0]) / max(1e-6, _TIRE_HEIGHT[-1] - _TIRE_HEIGHT[0])
        )

        # 1) 地面（轮胎阵是平地上摆轮胎，不挖坑——挖坑就没法从旁边绕了）
        for box in make_plane(body, self.size, 0.0, center_zero=False):
            geometries.append(TerrainGeometry(geom=box, color=_GROUND_COLOR))

        # 2) 轮胎阵列：每个轮胎 = ring_balls 个小圆柱围成环
        x0 = self.start_length + 1.2
        ring_r = self.ring_radius
        ball_r = self.ball_radius
        color = brand_ramp(_TIRE_COLOR[:3], d)
        for ix in range(self.num_x):
            for iy in range(self.num_y):
                cx = x0 + ix * self.pitch
                cyy = cy + (iy - (self.num_y - 1) / 2.0) * self.pitch
                for k in range(self.ring_balls):
                    ang = 2.0 * np.pi * k / self.ring_balls
                    px = cx + ring_r * np.cos(ang)
                    py = cyy + ring_r * np.sin(ang)
                    geom = body.add_geom(
                        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                        size=(ball_r, h / 2.0),
                        pos=(px, py, h / 2.0),
                    )
                    geometries.append(TerrainGeometry(geom=geom, color=color))

        origin = np.array([self.start_length / 2.0, cy, 0.0])
        return TerrainOutput(origin=origin, geometries=geometries)