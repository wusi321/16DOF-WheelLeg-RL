"""打开本地 MuJoCo 窗口，看实机标定站姿（带重力、PD 保持）。

这是最省事的一个「打开仿真看看」的脚本：不需要训练好的策略、不需要 checkpoint，
直接起一个 ``mujoco.viewer`` 窗口，机器人用标称站姿落稳后原地保持，
鼠标左键拖动旋转、右键平移、滚轮缩放。

用法::

    # 最简单：直接双击 / 命令行跑
    python tools\\view_stand.py

    # 一边看一边让它以 0.6 m/s 往前滚（四轮速度伺服给目标轮速）
    python tools\\view_stand.py --wheel-speed 6.0

    # 让它用「比站姿高一点」的姿态站着（把 thigh 目标整体抬高 0.1 rad）
    python tools\\view_stand.py --thigh-offset 0.1

按 ``Esc`` 或关窗口退出。
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from robot import naming  # noqa: E402
from robot.robot_cfg import get_spec  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="用 MuJoCo 窗口看 16DOF 轮足狗的站立姿态")
    parser.add_argument(
        "--wheel-speed",
        type=float,
        default=0.0,
        help="轮子目标角速度 (rad/s)，正数前进；默认 0（原地站着）",
    )
    parser.add_argument(
        "--thigh-offset",
        type=float,
        default=0.0,
        help="大腿目标角整体加一个偏置 (rad)，正数会让机器人站得更矮/更高，用来试姿态",
    )
    parser.add_argument(
        "--calf-offset",
        type=float,
        default=0.0,
        help="小腿目标角整体加一个偏置 (rad)",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=0.0,
        help="跑多少秒后自动退出（0 = 一直跑到关窗口，方便截图/录屏）",
    )
    return parser.parse_args()


def build_model() -> mujoco.MjModel:
    """机器人 spec + 一块临时地板（正式训练时地板由 mjlab 的地形实体提供）。"""
    spec = get_spec()
    floor = spec.worldbody.add_geom()
    floor.name = "viewer_floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [8.0, 8.0, 0.1]
    floor.rgba = [0.30, 0.32, 0.34, 1.0]
    return spec.compile()


def stand_targets(
    model: mujoco.MjModel, thigh_offset: float, calf_offset: float, wheel_speed: float
) -> tuple[dict[str, float], dict[str, float]]:
    """返回 (关节目标角, 轮子目标角速度)。"""
    targets: dict[str, float] = {}
    wheel_targets: dict[str, float] = {}
    for jid in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if name is None:
            continue
        for pattern, value in naming.STAND_JOINT_POS.items():
            if not re.fullmatch(pattern, name):
                continue
            if "wheel" in name:
                wheel_targets[name] = wheel_speed
            elif "thigh" in name:
                targets[name] = value + thigh_offset
            elif "calf" in name:
                targets[name] = value + calf_offset
            else:
                targets[name] = value
    return targets, wheel_targets


def apply_pd(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    targets: dict[str, float],
    wheel_targets: dict[str, float],
) -> None:
    """和训练里 BuiltinPosition/VelocityActuatorCfg 等效的 PD（作用在 qfrc_applied）。"""
    for name, target in targets.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        dof = model.jnt_dofadr[jid]
        q = float(data.qpos[model.jnt_qposadr[jid]])
        qd = float(data.qvel[dof])
        if "calf" in name:
            tau = naming.STIFFNESS_CALF * (target - q) - naming.DAMPING_CALF * qd
            limit = naming.EFFORT_CALF
        elif "thigh" in name:
            tau = naming.STIFFNESS_LEG * (target - q) - naming.DAMPING_LEG * qd
            limit = naming.EFFORT_THIGH
        else:
            tau = naming.STIFFNESS_LEG * (target - q) - naming.DAMPING_LEG * qd
            limit = naming.EFFORT_HIP
        data.qfrc_applied[dof] = float(np.clip(tau, -limit, limit))

    for name, target in wheel_targets.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        dof = model.jnt_dofadr[jid]
        qd = float(data.qvel[dof])
        tau = naming.DAMPING_WHEEL * (target - qd)
        data.qfrc_applied[dof] = float(np.clip(tau, -naming.EFFORT_WHEEL, naming.EFFORT_WHEEL))


def main() -> None:
    args = parse_args()
    model = build_model()
    data = mujoco.MjData(model)
    targets, wheel_targets = stand_targets(
        model, args.thigh_offset, args.calf_offset, args.wheel_speed
    )

    # 初始状态：直接摆到目标角 + 站在标称高度
    data.qpos[:] = 0.0
    for name, value in targets.items():
        data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]] = value
    root = model.jnt_qposadr[0]
    data.qpos[root : root + 3] = (0.0, 0.0, naming.STAND_BASE_HEIGHT)
    data.qpos[root + 3 : root + 7] = (1.0, 0.0, 0.0, 0.0)
    mujoco.mj_forward(model, data)

    print(__doc__)
    print(f"站姿目标: base_z = {naming.STAND_BASE_HEIGHT:.4f} m, 轮速 = {args.wheel_speed} rad/s")
    print("窗口已打开，拖动鼠标旋转视角；关窗口或按 Esc 退出。\n")

    base = model.body("base").id
    dt = model.opt.timestep
    sim_steps_per_sync = 4  # 每 4 个 0.002 s 步同步一次画面（≈80 Hz 刷新率足够顺）
    wall_per_sync = dt * sim_steps_per_sync
    start = time.perf_counter()
    next_sync = start

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = (0.0, 0.0, 0.26)
        viewer.cam.distance = 1.4
        viewer.cam.elevation = -12.0
        viewer.cam.azimuth = 135.0
        while viewer.is_running():
            apply_pd(model, data, targets, wheel_targets)
            mujoco.mj_step(model, data)
            if data.ncon and (data.qpos[root + 2] < 0.12):
                print(f"[WARN] 机身掉到 {data.qpos[root + 2]:.3f} m，可能摔了")
                break
            viewer.sync()
            next_sync += wall_per_sync
            sleep_for = next_sync - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_sync = time.perf_counter()  # 掉帧就重置节拍，别追帧
            if args.seconds > 0 and time.perf_counter() - start > args.seconds:
                break

    print(f"退出。最终 base_z = {float(data.xpos[base, 2]):.4f} m")


if __name__ == "__main__":
    main()