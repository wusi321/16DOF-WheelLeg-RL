"""Competition obstacle terrains for locomotion tasks.

Implemented in accordance with the interface specifications in
mjlab/src/mjlab/terrains/primitive_terrains.py. Terrain difficulty (difficulty in [0, 1])
is automatically supplied via TerrainGeneratorCfg and curriculum.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeometry, TerrainOutput
from mjlab.terrains.utils import make_plane
from mjlab.terrains import BoxPyramidStairsTerrainCfg
from mjlab.utils.color import brand_ramp

# Color constants matching primitive_terrains.py style
_COLOR_ORANGE = (0.95, 0.55, 0.10)
_COLOR_PURPLE = (0.60, 0.20, 0.80)
_COLOR_GREEN_RC = (0.13, 0.55, 0.24)  # 绿色（限高框架，官方规则要求"颜色为绿色"）


@dataclass(kw_only=True)
class RCWallTerrainCfg(SubTerrainCfg):
    """Repeated transverse wall obstacle terrain for high-wall gait training.

    The robot must repeatedly step/vault over transverse walls and proceed.
    As difficulty scales from 0 to 1, the wall height increases linearly from
    wall_height_range[0] to wall_height_range[1].
    
    Official spec (rules table 1): height 300mm, width 800mm, thickness 80mm.

    Difficulty scales the wall height from 0 to the official 300mm; width and
    thickness stay at the official values (they are not what makes it hard).
    """

    wall_height_range: tuple[float, float] = (0.0, 0.30)
    """Wall height range (m). Official: 300mm."""
    wall_thickness: float = 0.08
    """Wall thickness (m). Official: 80mm."""
    wall_length_frac: float = 0.5
    """Wall width as a fraction of terrain y-size. Official 800mm; on a 1.6m site 0.5 = 800mm."""
    platform_width: float = 1.5
    """Nominal start platform width (m)."""
    wall_centers_x: tuple[float, ...] = (2.1, 3.2, 4.3, 5.4, 6.5)
    """Wall center positions along x."""

    def function(
        self,
        difficulty: float,
        spec: mujoco.MjSpec,
        rng: np.random.Generator,
    ) -> TerrainOutput:
        del rng
        body = spec.body("terrain")
        geometries: list[TerrainGeometry] = []

        wall_height = self.wall_height_range[0] + difficulty * (
            self.wall_height_range[1] - self.wall_height_range[0]
        )

        # -- Ground floor base plane --
        floor_boxes = make_plane(body, self.size, 0.0, center_zero=False)
        floor_color = (0.45, 0.45, 0.45, 1.0)
        for box in floor_boxes:
            geometries.append(TerrainGeometry(geom=box, color=floor_color))

        if wall_height < 1e-3:
            origin = np.array([self.size[0] / 2, self.size[1] / 2, 0.0])
            return TerrainOutput(origin=origin, geometries=geometries)

        # -- Wall geometry: transverse walls oriented along y-axis --
        wall_length = self.wall_length_frac * self.size[1]
        cy = self.size[1] / 2

        wall_color = brand_ramp(_COLOR_ORANGE, difficulty)

        for cx in self.wall_centers_x:
            wall_geom = body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=(
                    self.wall_thickness / 2,   # half-size x
                    wall_length / 2,           # half-size y
                    wall_height / 2,           # half-size z
                ),
                pos=(cx, cy, wall_height / 2),
            )
            geometries.append(TerrainGeometry(geom=wall_geom, color=wall_color))

        # Spawn origin is set to the left platform area.
        origin = np.array([1.5, cy, 0.0])
        return TerrainOutput(origin=origin, geometries=geometries)


@dataclass(kw_only=True)
class RCLowBarTerrainCfg(SubTerrainCfg):
    """限高框架（官方规则表 1）——2026-11-28 按官方规格重写。

    **官方规格**：
        限高框架内高 290mm，内宽 800mm，长 800mm。限高框架颜色为绿色。

    **通过定义**（表 2）：
        机器人以任何运动姿态从限高框架下穿过（允许触碰，但不允许暴力撞击，
        例如导致框架出现移位），则表示限高框架障碍通过完成。

    几何（官方模型 Robocon2027.xml 实测）：
        4 根立柱 size=(0.02, 0.02, 0.145)，z=0.145  → 柱高 0.29
        4 根顶梁 z=0.31，半高 0.02                → 梁底 0.29（正好是内高）
        外沿 0.84×0.84，内空 0.80×0.80
        另有 1 根中间横梁 + 2 根中间立柱，都在顶部（加强用，不挡通行）

    旧版做的是"一根圆柱横杆 + 两个支座"，跟官方"方框"完全不是一回事，
    而且内高写成 0.25~0.35、宽度写成 1.0m，都已废弃。
    """

    clearance_range: tuple[float, float] = (0.29, 0.36)
    """内高范围（m）。难度 0 → 0.36（好钻），难度 1 → 0.29（官方标准）。"""

    inner_width: float = 0.80
    """内宽（m）。官方 800mm。"""

    inner_length: float = 0.80
    """内长（m）。官方 800mm。"""

    post_half: float = 0.02
    """立柱/横梁的截面半宽（m）。XML 实测 20mm。"""

    def function(
        self,
        difficulty: float,
        spec: mujoco.MjSpec,
        rng: np.random.Generator,
    ) -> TerrainOutput:
        del rng
        body = spec.body("terrain")
        geometries: list[TerrainGeometry] = []

        clearance = self.clearance_range[1] - difficulty * (
            self.clearance_range[1] - self.clearance_range[0]
        )
        cx = self.size[0] / 2.0
        cy = self.size[1] / 2.0
        ph = self.post_half
        half_w = self.inner_width / 2.0 + ph      # 外沿半宽
        half_l = self.inner_length / 2.0 + ph
        color = brand_ramp(_COLOR_GREEN_RC, difficulty)

        # -- Ground floor --
        floor_color = (0.45, 0.45, 0.45, 1.0)
        for box in make_plane(body, self.size, 0.0, center_zero=False):
            geometries.append(TerrainGeometry(geom=box, color=floor_color))

        # -- 4 根立柱（顶面 = 内高 clearance）--
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                post = body.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=(ph, ph, clearance / 2.0),
                    pos=(cx + sx * half_l, cy + sy * half_w, clearance / 2.0),
                )
                geometries.append(TerrainGeometry(geom=post, color=color))

        # -- 顶部横梁：梁底 = clearance（这样内高正好是 clearance）--
        beam_z = clearance + ph
        for sx in (-1.0, 1.0):
            beam = body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=(ph, half_w, ph),
                pos=(cx + sx * half_l, cy, beam_z),
            )
            geometries.append(TerrainGeometry(geom=beam, color=color))
        for sy in (-1.0, 1.0):
            beam = body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=(half_l, ph, ph),
                pos=(cx, cy + sy * half_w, beam_z),
            )
            geometries.append(TerrainGeometry(geom=beam, color=color))

        # -- 中间加强梁 + 中间立柱（XML 里也有，都在顶部，不影响通行）--
        mid_beam = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(ph, max(1e-3, half_w - 2 * ph), ph),
            pos=(cx, cy, beam_z),
        )
        geometries.append(TerrainGeometry(geom=mid_beam, color=color))
        for sy in (-1.0, 1.0):
            mid_post = body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=(ph, ph, clearance / 2.0),
                pos=(cx, cy + sy * half_w, clearance / 2.0),
            )
            geometries.append(TerrainGeometry(geom=mid_post, color=color))

        # 起点放在框架前 1m，让机器人有距离加速并调整姿态
        origin = np.array([cx - half_l - 1.0, cy, 0.0])
        return TerrainOutput(origin=origin, geometries=geometries)


@dataclass(kw_only=True)
class RCPyramidStairsTerrainCfg(BoxPyramidStairsTerrainCfg):
    """Refactored pyramid stairs terrain with the spawn origin shifted to the left edge.

    This ensures the robot can sprint/accelerate on flat ground before ascending,
    substantially reducing early falls or flips on vertical steps.
    """

    def function(
        self,
        difficulty: float,
        spec: mujoco.MjSpec,
        rng: np.random.Generator,
    ) -> TerrainOutput:
        output = super().function(difficulty, spec, rng)
        cy = self.size[1] / 2
        # Relocate the spawn origin to the left platform zone
        output.origin = np.array([1.5, cy, 0.0])
        return output
