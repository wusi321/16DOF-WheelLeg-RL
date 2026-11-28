"""Step 1: 45-degree standing-pose gate.

Builds the robot spec through ``robot.robot_cfg.get_spec()`` (which applies the
self-collision fix), adds a floor and the calibrated PD actuators, puts the
robot in the user-specified stance, then runs zero action for a few seconds.

Pass criteria:
  * initial kinematic height ~= 0.3362 m
  * final base height in [0.31, 0.35] m
  * upright (up_z > 0.99)
  * no self collision at t=0 and at the end
  * four wheels touching the floor
  * no NaN, leg joint tracking error < 0.10 rad
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from robot import naming, robot_cfg  # noqa: E402

FLOOR_NAME = "floor"


def _add_floor(spec: mujoco.MjSpec) -> None:
    spec.worldbody.add_geom(
        name=FLOOR_NAME,
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        pos=[0.0, 0.0, 0.0],
        size=[0.0, 0.0, 0.05],
        friction=[1.0, 0.005, 0.0001],
        rgba=[0.35, 0.35, 0.35, 1.0],
    )


def _add_position_actuator(
    spec: mujoco.MjSpec,
    joint_name: str,
    kp: float,
    kv: float,
    effort: float,
) -> None:
    joint = spec.joint(joint_name)
    lo, hi = float(joint.range[0]), float(joint.range[1])
    spec.add_actuator(
        name=f"{joint_name}_pos",
        target=joint_name,
        trntype=mujoco.mjtTrn.mjTRN_JOINT,
        gaintype=mujoco.mjtGain.mjGAIN_FIXED,
        gainprm=[kp, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        biastype=mujoco.mjtBias.mjBIAS_AFFINE,
        biasprm=[0, -kp, -kv, 0, 0, 0, 0, 0, 0, 0],
        ctrllimited=1,
        ctrlrange=[lo, hi],
        forcelimited=1,
        forcerange=[-effort, effort],
    )


def _add_velocity_actuator(
    spec: mujoco.MjSpec,
    joint_name: str,
    kv: float,
    effort: float,
    ctrl_limit: float = 30.0,
) -> None:
    spec.add_actuator(
        name=f"{joint_name}_vel",
        target=joint_name,
        trntype=mujoco.mjtTrn.mjTRN_JOINT,
        gaintype=mujoco.mjtGain.mjGAIN_FIXED,
        gainprm=[kv, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        biastype=mujoco.mjtBias.mjBIAS_AFFINE,
        biasprm=[0, 0, -kv, 0, 0, 0, 0, 0, 0, 0],
        ctrllimited=1,
        ctrlrange=[-ctrl_limit, ctrl_limit],
        forcelimited=1,
        forcerange=[-effort, effort],
    )


def _add_actuators(spec: mujoco.MjSpec) -> None:
    for leg in naming.LEG_ORDER:
        for part in ("hip", "thigh"):
            _add_position_actuator(
                spec,
                f"{leg}_{part}_joint",
                naming.STIFFNESS_LEG,
                naming.DAMPING_LEG,
                naming.EFFORT_HIP if part == "hip" else naming.EFFORT_THIGH,
            )
        _add_position_actuator(
            spec,
            f"{leg}_calf_joint",
            naming.STIFFNESS_CALF,
            naming.DAMPING_CALF,
            naming.EFFORT_CALF,
        )
        _add_velocity_actuator(
            spec,
            f"{leg}_wheel_joint",
            naming.DAMPING_WHEEL,
            naming.EFFORT_WHEEL,
        )


def _contact_stats(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[int, int, int, list[str]]:
    """Return (floor contacts, self contacts, wheel-ground geoms, self pairs)."""
    floor_contacts = 0
    self_contacts = 0
    wheel_geoms: set[str] = set()
    self_pairs: list[str] = []
    for i in range(data.ncon):
        contact = data.contact[i]
        g1 = model.geom(contact.geom1).name
        g2 = model.geom(contact.geom2).name
        if g1 == FLOOR_NAME or g2 == FLOOR_NAME:
            floor_contacts += 1
            other = g2 if g1 == FLOOR_NAME else g1
            if "wheel" in other:
                wheel_geoms.add(other)
        else:
            self_contacts += 1
            self_pairs.append(f"{g1}<->{g2}")
    return floor_contacts, self_contacts, len(wheel_geoms), self_pairs


def _base_height(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    return float(data.xpos[model.body(naming.BASE_BODY).id][2])


def _up_z(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    rot = np.asarray(data.xmat[model.body(naming.BASE_BODY).id]).reshape(3, 3)
    return float(rot[2, 2])


def _max_leg_error(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    error = 0.0
    for name, target in naming.STAND_JOINT_POS_BY_NAME.items():
        if name.endswith("_wheel_joint"):
            continue
        jid = model.joint(name).id
        q = float(data.qpos[model.jnt_qposadr[jid]])
        error = max(error, abs(q - target))
    return error


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--timestep", type=float, default=0.005)
    parser.add_argument("--height", type=float, default=naming.STAND_BASE_HEIGHT)
    args = parser.parse_args()

    print("=" * 78)
    print("Step 1: 45-degree standing-pose gate")
    print("=" * 78)
    print(f"xml       : {robot_cfg.ROBOT_XML}")
    print(f"height    : {args.height:.4f} m")
    print(f"timestep  : {args.timestep:.4f} s, duration {args.seconds:.1f} s")
    print()

    spec = robot_cfg.get_spec()
    _add_floor(spec)
    _add_actuators(spec)
    spec.option.timestep = args.timestep
    model = spec.compile()
    data = mujoco.MjData(model)

    if (model.nq, model.nu) != (23, 16):
        print(f"[FAIL] nq/nu = {model.nq}/{model.nu}, expected 23/16")
        return 1

    mujoco.mj_resetData(model, data)
    free_jid = model.joint("root_free").id
    free_adr = int(model.jnt_qposadr[free_jid])
    data.qpos[free_adr : free_adr + 3] = [0.0, 0.0, args.height]
    data.qpos[free_adr + 3 : free_adr + 7] = [1.0, 0.0, 0.0, 0.0]
    for name, angle in naming.STAND_JOINT_POS_BY_NAME.items():
        jid = model.joint(name).id
        data.qpos[model.jnt_qposadr[jid]] = angle
    data.qvel[:] = 0.0

    for aid in range(model.nu):
        jid = int(model.actuator_trnid[aid, 0])
        jname = model.joint(jid).name
        data.ctrl[aid] = 0.0 if jname.endswith("_wheel_joint") else naming.STAND_JOINT_POS_BY_NAME[jname]

    mujoco.mj_forward(model, data)
    z0 = _base_height(model, data)
    up0 = _up_z(model, data)
    floor0, self0, wheel0, self_pairs0 = _contact_stats(model, data)

    print(f"t=0: base_z={z0:.4f} m  up_z={up0:.4f}  floor={floor0}  self={self0}  wheels={wheel0}")
    if self0:
        print("  self contacts:")
        for pair in self_pairs0:
            print(f"    {pair}")
    print()
    print(f"{'t(s)':>6} {'base_z':>8} {'up_z':>8} {'floor':>6} {'self':>5} {'wheel':>6} {'jerr':>7}")
    print("-" * 78)

    dt = float(model.opt.timestep)
    n_steps = int(round(args.seconds / dt))
    record_every = max(1, int(round(0.5 / dt)))
    records = []

    for step in range(n_steps + 1):
        if step % record_every == 0 or step == n_steps:
            z = _base_height(model, data)
            up = _up_z(model, data)
            floor, self_c, wheel, _ = _contact_stats(model, data)
            jerr = _max_leg_error(model, data)
            records.append((step * dt, z, up, floor, self_c, wheel, jerr))
            print(
                f"{step * dt:6.2f} {z:8.4f} {up:8.4f} {floor:6d} {self_c:5d} {wheel:6d} {jerr:7.4f}"
            )
        if step == n_steps:
            break
        mujoco.mj_step(model, data)

    t_end, z_end, up_end, floor_end, self_end, wheel_end, jerr_end = records[-1]
    finite = bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())

    checks = [
        ("initial height ~= 0.3362", abs(z0 - naming.STAND_BASE_HEIGHT) < 0.005, f"{z0:.4f}"),
        ("no self collision at t=0", self0 == 0, f"{self0}"),
        ("final height in [0.31, 0.35]", 0.31 <= z_end <= 0.35, f"{z_end:.4f}"),
        ("upright up_z > 0.99", up_end > 0.99, f"{up_end:.4f}"),
        ("no self collision at end", self_end == 0, f"{self_end}"),
        ("4 wheels on floor at end", wheel_end >= 4, f"{wheel_end}"),
        ("finite state", finite, str(finite)),
        ("leg joint error < 0.10 rad", jerr_end < 0.10, f"{jerr_end:.4f}"),
    ]

    print()
    print("checks:")
    ok = True
    for label, passed, detail in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label} ({detail})")
        ok = ok and passed

    print()
    if ok:
        print("[PASS] standing-pose gate passed.")
        return 0
    print("[FAIL] standing-pose gate failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())