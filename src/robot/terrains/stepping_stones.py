"""梅花桩（stepping-stone）地形，对应 Robocon2027.xml 里的 stepping_stones 段。

实场地形（`Robocon2027.xml`）：

    stepping_stones_01_00..05   size = 0.1 0.1 0.1（即 0.20 x 0.20 x 0.20 m）
    z 中心 0.10 -> 顶面 0.20 m
    两条桩列：y = 2.2 / 1.8（横向中心距 0.40 m）
    同列前后中心距 0.60 m，两条桩列在 x 方向错开 0.30 m

轮距 0.385 m ≈ 桩列间距 0.40 m，所以实车的左右轮分别落在两条桩列上；
0.60 m 的中心距大于轴距 0.387 m，必须靠抬腿跨越，不能纯滚动通过。

训练用地形把这一布局参数化，难度 d ∈ [0, 1] 沿桩高 / 中心距 / 错位 / 位置噪声 /
缺失桩逐级放宽（对应 16DOF_RL.md 的六个阶段）：

    d ≈ 0.00 - 0.15  阶段 1：桩高 0.06~0.08 m，中心距 0.30 m，无错位、无噪声
    d ≈ 0.15 - 0.30  阶段 2：桩高 ~0.10 m，固定布局
    d ≈ 0.30 - 0.50  阶段 3：桩高升到 0.15 m，中心距/错位小幅随机
    d ≈ 0.50 - 0.75  阶段 4：桩高到 0.18 m，中心距到 0.55 m，开始少量缺桩
    d ≈ 0.75 - 1.00  阶段 5：桩高 0.20 m、中心距 0.60 m、错位 0.30 m，
                     位置噪声 ±0.03 m、缺桩 5%~10% —— 与实场一致
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeometry, TerrainOutput
from mjlab.terrains.utils import make_plane
from mjlab.utils.color import brand_ramp

#: 课程控制点：难度 -> 参数（线性插值）。
_STAGE_DIFFICULTY = np.array([0.00, 0.15, 0.30, 0.50, 0.75, 1.00])
_HEIGHT_LOW = np.array([0.06, 0.08, 0.10, 0.10, 0.15, 0.18])
_HEIGHT_HIGH = np.array([0.08, 0.10, 0.10, 0.15, 0.18, 0.20])
_PITCH = np.array([0.30, 0.32, 0.35, 0.45, 0.55, 0.60])
_STAGGER = np.array([0.00, 0.05, 0.10, 0.20, 0.26, 0.30])
_NOISE = np.array([0.00, 0.00, 0.01, 0.02, 0.025, 0.030])
_MISSING = np.array([0.00, 0.00, 0.00, 0.05, 0.075, 0.10])

_STONE_COLOR = (0.62, 0.52, 0.38)   # 木桩
_PLATFORM_COLOR = (0.45, 0.45, 0.45, 1.0)
_GOAL_COLOR = (0.95, 0.55, 0.10, 1.0)


@dataclass(kw_only=True)
class BoxSteppingStonesTerrainCfg(SubTerrainCfg):
    """两条交错桩列的梅花桩地形。

    每个 patch 沿 +x 方向通过：起点平台 -> 桩区 -> 终点平台。桩区左右各一条桩列，
    横向中心距默认 0.40 m（与实场一致、约等于轮距）。
    """

    stone_size: tuple[float, float] = (0.20, 0.20)
    """桩体平面尺寸 (x, y)，单位 m。"""

    lane_spacing: float = 0.40
    """左右两条桩列的横向中心距，单位 m。"""

    start_length: float = 0.8
    """起点平台长度，单位 m。"""

    end_length: float = 0.8
    """终点平台长度，单位 m。"""

    pit_depth: float = 0.6
    """桩区外侧/桩间地面相对桩顶平台的下沉深度，单位 m。

    0.6 m 时桩间是深坑：掉下去会触发 base_ground_contact / base_too_low 终止，
    逼策略真正踩在桩上；设为 0.0 则与实场一致（桩立在地面上）。
    """

    stone_height_range: tuple[float, float] = (0.06, 0.20)
    """桩高范围，仅用于文档/校验：实际曲线见模块顶部 _HEIGHT_* 控制点。"""

    pitch_range: tuple[float, float] = (0.30, 0.60)
    """前后中心距范围，仅用于文档/校验（实际曲线见 _PITCH）。"""

    stagger_range: tuple[float, float] = (0.0, 0.30)
    """两条桩列的 x 向错位范围，仅用于文档/校验（实际曲线见 _STAGGER）。"""

    position_noise_range: tuple[float, float] = (0.0, 0.03)
    """单桩位置扰动范围，仅用于文档/校验（实际曲线见 _NOISE）。"""

    missing_prob_range: tuple[float, float] = (0.0, 0.10)
    """单桩缺失概率范围，仅用于文档/校验（实际曲线见 _MISSING）。"""

    # -- 内部工具 -----------------------------------------------------------
    def _params(self, difficulty: float) -> dict[str, float]:
        d = float(np.clip(difficulty, 0.0, 1.0))
        return {
            "height_low": float(np.interp(d, _STAGE_DIFFICULTY, _HEIGHT_LOW)),
            "height_high": float(np.interp(d, _STAGE_DIFFICULTY, _HEIGHT_HIGH)),
            "pitch": float(np.interp(d, _STAGE_DIFFICULTY, _PITCH)),
            "stagger": float(np.interp(d, _STAGE_DIFFICULTY, _STAGGER)),
            "noise": float(np.interp(d, _STAGE_DIFFICULTY, _NOISE)),
            "missing": float(np.interp(d, _STAGE_DIFFICULTY, _MISSING)),
        }

    @staticmethod
    def stage_of(difficulty: float) -> int:
        """把难度映射成 1..5 的阶段编号（用于日志/调试）。"""
        d = float(np.clip(difficulty, 0.0, 1.0))
        if d < 0.15:
            return 1
        if d < 0.30:
            return 2
        if d < 0.50:
            return 3
        if d < 0.75:
            return 4
        return 5

    # -- 生成 ---------------------------------------------------------------
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
        p = self._params(difficulty)
        half_x, half_y = self.stone_size[0] / 2.0, self.stone_size[1] / 2.0

        # 1) 桩区地面：默认做成深坑，避免策略从地面绕过去
        if self.pit_depth > 0.0:
            floor = body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=(sx / 2.0, sy / 2.0, 0.05),
                pos=(sx / 2.0, cy, -self.pit_depth - 0.05),
            )
            geometries.append(TerrainGeometry(geom=floor, color=(0.18, 0.18, 0.20, 1.0)))
        else:
            for box in make_plane(body, self.size, 0.0, center_zero=False):
                geometries.append(TerrainGeometry(geom=box, color=(0.35, 0.35, 0.35, 1.0)))

        # 2) 起点 / 终点平台（顶面 z = 0）
        start = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(self.start_length / 2.0, sy / 2.0, 0.05),
            pos=(self.start_length / 2.0, cy, -0.05),
        )
        geometries.append(TerrainGeometry(geom=start, color=_PLATFORM_COLOR))
        end = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(self.end_length / 2.0, sy / 2.0, 0.05),
            pos=(sx - self.end_length / 2.0, cy, -0.05),
        )
        geometries.append(TerrainGeometry(geom=end, color=_PLATFORM_COLOR))

        # 3) 终点标记（纯视觉，不参与碰撞）
        goal_marker = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(0.06, sy / 2.0 - 0.05, 0.002),
            pos=(sx - self.end_length + 0.06, cy, 0.002),
        )
        goal_marker.contype = 0
        goal_marker.conaffinity = 0
        geometries.append(TerrainGeometry(geom=goal_marker, color=_GOAL_COLOR))

        # 4) 两条桩列
        pitch = p["pitch"]
        stagger = p["stagger"]
        noise = p["noise"]
        p_missing = p["missing"]

        x_first = self.start_length + pitch / 2.0
        x_last = sx - self.end_length - pitch / 2.0
        num_rows = max(1, int(np.floor((x_last - x_first) / pitch)) + 1)
        lane_offset = self.lane_spacing / 2.0
        x_margin = half_x + 0.02
        y_margin = half_y + 0.02

        stone_color = brand_ramp(_STONE_COLOR, 0.25 + 0.55 * float(np.clip(difficulty, 0.0, 1.0)))

        for k in range(num_rows):
            # 同一排不允许两条桩列同时缺失，否则地形可能不可通行
            drop = [bool(rng.random() < p_missing) for _ in range(2)]
            if drop[0] and drop[1]:
                drop[int(rng.integers(0, 2))] = False

            for lane in (0, 1):
                if drop[lane]:
                    continue

                x = x_first + k * pitch + (stagger if lane == 1 else 0.0)
                y = cy + (lane_offset if lane == 1 else -lane_offset)
                if noise > 0.0:
                    x += float(rng.uniform(-noise, noise))
                    y += float(rng.uniform(-noise, noise))

                # 边界检查：超界的桩直接丢弃，避免插入平台或 patch 外
                if x - x_margin < self.start_length or x + x_margin > sx - self.end_length:
                    continue
                if y - y_margin < 0.0 or y + y_margin > sy:
                    continue

                height = float(rng.uniform(p["height_low"], p["height_high"]))
                # 桩从坑底一直长到目标高度：既符合“实心木桩”，也防止从下方穿过
                bottom = -self.pit_depth
                half_h = (height - bottom) / 2.0
                stone = body.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=(half_x, half_y, half_h),
                    pos=(x, y, bottom + half_h),
                )
                geometries.append(
                    TerrainGeometry(geom=stone, color=stone_color)
                )

        origin = np.array([self.start_length / 2.0, cy, 0.0])
        return TerrainOutput(origin=origin, geometries=geometries)
