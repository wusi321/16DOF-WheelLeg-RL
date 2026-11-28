"""16DOF-WheelLeg 样机在 mjlab 里的装配层。

物理模型直接读 ``16DOF-URDF/mujoco/16DOF-URDF-training.xml``（box/cylinder
图元碰撞的训练模型）。本文件只负责：

1. **清理 MJCF**：删掉自带 floor（地形由 mjlab terrain 实体提供）、删掉自带
   16 个 actuator（改成下面的 BuiltinPosition/Velocity 组）、清掉 geom.margin
   （mujoco-warp 的 MULTICCD 要求 margin=0）。
2. **定义执行器**：12 个腿部关节位置控制 + 4 个轮子速度控制，增益/力矩按本机
   标定值。
3. **定义初始状态和动作缩放**：标称 45° 站姿、base 初始高度、动作系数。
"""

from __future__ import annotations

import functools
from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg, BuiltinVelocityActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from . import naming

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = _PROJECT_ROOT / "16DOF-URDF"
ROBOT_XML = MODEL_ROOT / "mujoco" / "16DOF-URDF-training.xml"

#: 机身碰撞盒的 y 半宽（原模型 0.1384），收窄到贴着实机骨架。
BASE_COLLISION_HALF_WIDTH = 0.125
#: 大腿碰撞圆柱向外移的横向距离（贴合机身外侧的腿板）。
THIGH_COLLISION_OUTWARD_SHIFT = 0.035
#: 允许读入的网格后缀（assets 只放网格，不放 XML）。
_MESH_SUFFIXES = {".stl", ".obj", ".msh"}


# ---------------------------------------------------------------------------
# 模型加载
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def _mesh_assets() -> dict[str, bytes]:
    """把 MJCF 用到的 STL 全部读进内存。

    新模型的 XML 在 ``16DOF-URDF/mujoco/``，网格有两套相对路径：
    ``split/...``（相对 mujoco/）和 ``../meshes/...``（相对 mujoco/ 的上一级）。
    同时提供裸文件名兜底。这样 ``MjSpec.from_string`` 完全不访问磁盘，也避开
    Windows 上非 ASCII 路径的问题。
    """
    assets: dict[str, bytes] = {}
    search_roots = (
        (ROBOT_XML.parent, ""),  # split/..., collision_hulls/...
        (MODEL_ROOT / "meshes", "../meshes/"),  # ../meshes/...
    )
    for root, prefix in search_roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in _MESH_SUFFIXES:
                continue
            data = path.read_bytes()
            rel = path.relative_to(root).as_posix()
            assets[f"{prefix}{rel}"] = data
            assets[rel] = data  # 相对 URDF 根（meshes/...）
            assets[path.name] = data  # 裸文件名兜底
    return assets


def _name_geoms(spec: mujoco.MjSpec) -> None:
    """给没名字的 geom 起名：``<body>_visual`` / ``<body>_collision``（重复加序号）。

    新 XML 的碰撞 geom 已经有名字，视觉 geom 仍然匿名；补名字后 mjlab 的域随机化
    和调试面板才能用正则匹配。
    """
    used: dict[str, int] = {}
    for geom in spec.geoms:
        if geom.name:
            continue
        body = geom.parent.name if geom.parent is not None else "world"
        kind = "visual" if geom.group == 2 else "collision"
        base = f"{body}_{kind}"
        count = used.get(base, 0)
        used[base] = count + 1
        geom.name = base if count == 0 else f"{base}_{count}"


def _slim_self_collision(spec: mujoco.MjSpec) -> None:
    """收窄机身碰撞盒、外移大腿碰撞圆柱，消除标称 45° 站姿的假自碰撞。

    实测原始模型在 45° 站姿下有 7 个 ``*_thigh_Link <-> base`` 接触，穿透
    3.2–3.4 cm，会把机器人顶到 z≈0.23，站不到 0.336。不改关节和惯量，只做：

    * 机身碰撞盒 y 半宽 0.1384 -> 0.125；
    * 4 个大腿碰撞圆柱沿 y 向外移 0.035 m。

    改完后站姿 0 自碰撞，零动作 PD 稳定在 0.333 m。
    """
    for geom in spec.geoms:
        if geom.group == 2 or geom.parent is None:
            continue  # 只看碰撞体
        body = geom.parent.name
        if body == naming.BASE_BODY and geom.type == mujoco.mjtGeom.mjGEOM_BOX:
            geom.size[1] = BASE_COLLISION_HALF_WIDTH
        elif body.endswith("_thigh_Link") and geom.type in (
            mujoco.mjtGeom.mjGEOM_CYLINDER,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
        ):
            side = -1.0 if geom.pos[1] < 0.0 else 1.0
            geom.pos[1] += side * THIGH_COLLISION_OUTWARD_SHIFT


def get_spec() -> mujoco.MjSpec:
    """构造机器人 spec：去掉自带地面和电机，保留 mjlab 需要的碰撞/惯量。"""
    xml = ROBOT_XML.read_text(encoding="utf-8-sig")
    spec = mujoco.MjSpec.from_string(xml, assets=dict(_mesh_assets()))

    # 1) 地面由 mjlab terrain 实体生成，MJCF 自带的 floor 必须删掉。
    for geom in list(spec.geoms):
        if geom.name == "floor" or geom.type == mujoco.mjtGeom.mjGEOM_PLANE:
            spec.delete(geom)

    # 2) 清 margin：mujoco-warp 开 MULTICCD 时不允许非零 margin。
    for geom in spec.geoms:
        geom.margin = 0.0

    # 3) 删掉 MJCF 自带的 16 个电机，改由 ARTICULATION_CFG 定义。
    for actuator in list(spec.actuators):
        spec.delete(actuator)

    # 3b) 自带 keyframe 的 ctrl 长度跟着电机走，电机删掉后长度对不上；
    #     初始状态由 INIT_STATE 定义，直接删掉 keyframe。
    for key in list(spec.keys):
        spec.delete(key)

    # 4) 给匿名 geom 补名字。
    _name_geoms(spec)

    # 5) 修掉 CAD 粗碰撞体造成的标称站姿假自碰撞。
    _slim_self_collision(spec)

    return spec


# ---------------------------------------------------------------------------
# 执行器
# ---------------------------------------------------------------------------
HIP_THIGH_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    target_names_expr=naming.HIP_JOINT_PATTERNS + naming.THIGH_JOINT_PATTERNS,
    stiffness=naming.STIFFNESS_LEG,
    damping=naming.DAMPING_LEG,
    effort_limit=naming.EFFORT_HIP,
)

CALF_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    target_names_expr=naming.CALF_JOINT_PATTERNS,
    stiffness=naming.STIFFNESS_CALF,
    damping=naming.DAMPING_CALF,
    effort_limit=naming.EFFORT_CALF,
)

WHEEL_ACTUATOR_CFG = BuiltinVelocityActuatorCfg(
    target_names_expr=naming.WHEEL_JOINT_PATTERNS,
    damping=naming.DAMPING_WHEEL,
    effort_limit=naming.EFFORT_WHEEL,
)

ARTICULATION_CFG = EntityArticulationInfoCfg(
    actuators=(HIP_THIGH_ACTUATOR_CFG, CALF_ACTUATOR_CFG, WHEEL_ACTUATOR_CFG),
    soft_joint_pos_limit_factor=0.95,
)


# ---------------------------------------------------------------------------
# 初始状态
# ---------------------------------------------------------------------------
INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, naming.STAND_BASE_HEIGHT),
    joint_pos=dict(naming.STAND_JOINT_POS),
    joint_vel={".*": 0.0},
)


def get_robot_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=INIT_STATE,
        spec_fn=get_spec,
        articulation=ARTICULATION_CFG,
    )


# ---------------------------------------------------------------------------
# 动作缩放
# ---------------------------------------------------------------------------
#: 腿部位置目标：髋 ±0.125 rad，大腿/小腿 ±0.25 rad。
LEG_POS_SCALE = 0.25
#: 轮子速度目标：±10 rad/s（≈ ±0.84 m/s 线速度）。
WHEEL_VEL_SCALE = 10.0