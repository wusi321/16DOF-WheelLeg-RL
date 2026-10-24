"""Offline contract check for the go2w port.

Validates that every name pattern / pose / geometry constant used by the task
still matches the go2w MJCF.  Runs with nothing but ``mujoco`` installed:

    uv run --no-project --with mujoco python tools/check_go2w_contract.py

It stubs out ``mjlab`` so that ``src/robot/robot_cfg.py`` can be imported
without the full training stack (torch / warp / mujoco-warp).
"""

from __future__ import annotations

import importlib
import re
import sys
import types
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Minimal mjlab stubs: robot_cfg only needs these constructors at import time.
# ---------------------------------------------------------------------------
def _install_mjlab_stubs() -> None:
    mjlab = types.ModuleType("mjlab")
    actuator = types.ModuleType("mjlab.actuator")

    class _ActuatorCfg:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __repr__(self) -> str:
            return f"{type(self).__name__}({self.kwargs})"

    actuator.BuiltinPositionActuatorCfg = type(
        "BuiltinPositionActuatorCfg", (_ActuatorCfg,), {}
    )
    actuator.BuiltinVelocityActuatorCfg = type(
        "BuiltinVelocityActuatorCfg", (_ActuatorCfg,), {}
    )

    entity = types.ModuleType("mjlab.entity")

    class InitialStateCfg:
        def __init__(self, pos=(0, 0, 0), rot=(1, 0, 0, 0), joint_pos=None, joint_vel=None):
            self.pos = pos
            self.rot = rot
            self.joint_pos = joint_pos or {}
            self.joint_vel = joint_vel or {}

    class EntityCfg:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    EntityCfg.InitialStateCfg = InitialStateCfg

    class EntityArticulationInfoCfg:
        def __init__(self, actuators=(), soft_joint_pos_limit_factor=1.0):
            self.actuators = actuators
            self.soft_joint_pos_limit_factor = soft_joint_pos_limit_factor

    entity.EntityCfg = EntityCfg
    entity.EntityArticulationInfoCfg = EntityArticulationInfoCfg
    entity.InitialStateCfg = InitialStateCfg

    sys.modules.update(
        {
            "mjlab": mjlab,
            "mjlab.actuator": actuator,
            "mjlab.entity": entity,
        }
    )


# Import the `robot` package without executing src/robot/__init__.py (that one
# pulls in the whole mjlab env/registry stack).
pkg = types.ModuleType("robot")
pkg.__path__ = [str(ROOT / "src" / "robot")]
sys.modules["robot"] = pkg
_install_mjlab_stubs()

robot_cfg = importlib.import_module("robot.robot_cfg")
naming = importlib.import_module("robot.naming")

FAILS: list[str] = []


def check(condition: bool, message: str) -> None:
    status = "ok  " if condition else "FAIL"
    print(f"[{status}] {message}")
    if not condition:
        FAILS.append(message)


def matches(patterns, names) -> list[str]:
    out = []
    for name in names:
        if not name:
            continue
        if any(re.fullmatch(p, name) for p in patterns):
            out.append(name)
    return out


spec = robot_cfg.get_spec()
model = spec.compile()

joint_names = [
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)
]
body_names = [
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)
]

print(f"model: nq={model.nq} nv={model.nv} nu={model.nu} njnt={model.njnt} "
      f"nbody={model.nbody} nmesh={model.nmesh}")
print(f"actuators left in spec: {len(spec.actuators)} (expected 0)")

check(len(spec.actuators) == 0, "MJCF motors removed from the spec")

leg = matches(naming.LEG_JOINT_PATTERNS, joint_names)
wheel = matches(naming.WHEEL_JOINT_PATTERNS, joint_names)
check(len(leg) == 12, f"leg joint patterns match 12 joints (got {len(leg)}: {leg[:3]}...)")
check(len(wheel) == 4, f"wheel joint patterns match 4 joints (got {wheel})")

wheels = matches(naming.WHEEL_BODY_PATTERNS, body_names)
check(len(wheels) == 4, f"wheel body patterns match 4 bodies (got {wheels})")
collision_bodies = matches(naming.BODY_COLLISION_PATTERNS, body_names)
check(
    len(collision_bodies) == 12,
    f"body collision patterns match 12 leg bodies (got {len(collision_bodies)})",
)
check(naming.BASE_BODY in body_names, f"base body '{naming.BASE_BODY}' exists")

# The critic height scanner raycasts geom group 0 only (mjlab terrain lives in
# group 0), so no robot geom may stay in group 0 -- otherwise the scan hits the
# robot itself.
robot_group0 = []
wheel_visual_groups: dict[str, list[int]] = {}
for g in range(model.ngeom):
    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g])
    if not body_name or body_name == "world":
        continue
    if model.geom_group[g] == 0:
        robot_group0.append((body_name, g))
    if body_name.endswith("_wheel_link") and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
        wheel_visual_groups.setdefault(body_name, []).append(int(model.geom_group[g]))
check(not robot_group0, f"no robot geom left in group 0 (got {len(robot_group0)})")
check(
    all(2 in groups for groups in wheel_visual_groups.values()) and len(wheel_visual_groups) == 4,
    f"each wheel has a visual mesh in group 2 (got {wheel_visual_groups})",
)

# Actuator groups.
for cfg in (
    robot_cfg.HIP_THIGH_ACTUATOR_CFG,
    robot_cfg.CALF_ACTUATOR_CFG,
    robot_cfg.WHEEL_ACTUATOR_CFG,
):
    targets = matches(cfg.kwargs["target_names_expr"], joint_names)
    check(len(targets) > 0, f"actuator group {cfg.kwargs['target_names_expr']} -> {len(targets)} joints")

# Joint order used by observations / actions.  mjlab resolves names in MJCF
# order, so the wheel joint of each leg sits right after its calf joint
# (FL_hip, FL_thigh, FL_calf, FL_wheel, FR_...).  The action manager
# concatenates both action terms, so actions are [12 leg joints][4 wheels].
expected_joints = [
    f"{leg_}_{dof}_joint"
    for leg_ in ("FL", "FR", "RL", "RR")
    for dof in ("hip", "thigh", "calf", "wheel")
]
check(
    joint_names[1:17] == expected_joints,
    f"joint order is FL, FR, RL, RR x (hip, thigh, calf, wheel): {joint_names[1:17]}",
)

# Nominal poses must be inside the joint limits.
qpos_adr = {n: model.jnt_qposadr[i] for i, n in enumerate(joint_names) if n}
limits = {n: model.jnt_range[i] for i, n in enumerate(joint_names) if n}

for label, pose, height in (
    ("stand", naming.STAND_JOINT_POS, naming.STAND_BASE_HEIGHT),
    ("crawl", naming.CRAWL_JOINT_POS, naming.CRAWL_BASE_HEIGHT),
):
    for leg_ in naming.LEG_ORDER:
        for dof, value in pose.items():
            rng = limits[f"{leg_}_{dof}_joint"]
            inside = float(rng[0]) <= value <= float(rng[1])
            check(inside, f"{label}: {leg_}_{dof}_joint={value} inside range {rng}")

# Forward kinematics: wheel bottom should sit on the ground at the nominal height.
def wheel_bottom(pose: dict, hip_sign: float, height: float) -> float:
    data = mujoco.MjData(model)
    q = np.zeros(model.nq)
    q[3] = 1.0
    q[0:3] = (0.0, 0.0, height)
    for leg_ in naming.LEG_ORDER:
        q[qpos_adr[f"{leg_}_hip_joint"]] = hip_sign * pose["hip"]
        q[qpos_adr[f"{leg_}_thigh_joint"]] = pose["thigh"]
        q[qpos_adr[f"{leg_}_calf_joint"]] = pose["calf"]
    data.qpos = q
    mujoco.mj_forward(model, data)
    lowest = []
    for g in range(model.ngeom):
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g])
        if body and body.endswith("_wheel_link"):
            mid = model.geom_dataid[g]
            if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            adr, num = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            verts = model.mesh_vert[adr : adr + num]
            rot = data.geom_xmat[g].reshape(3, 3)
            world = verts @ rot.T + data.geom_xpos[g]
            lowest.append(world[:, 2].min())
    return float(np.mean(lowest))


for label, pose, height in (
    ("stand", naming.STAND_JOINT_POS, naming.STAND_BASE_HEIGHT),
    ("crawl", naming.CRAWL_JOINT_POS, naming.CRAWL_BASE_HEIGHT),
):
    for sign in (+1.0, -1.0):
        bottom = wheel_bottom(pose, sign, height)
        check(
            abs(bottom) < 0.01,
            f"{label} pose (hip sign {sign:+.0f}): wheel bottom at z={bottom:+.4f} m",
        )

print()
if FAILS:
    print(f"{len(FAILS)} check(s) failed")
    sys.exit(1)
print("all checks passed")
