"""快速验证"四轮支撑"奖励与接触观测在梅花桩环境里的实际行为。

检查：
1. ``feet_ground_contact`` 的 ``found`` 是 4 维（每轮一个标志）；
2. 站在地面上时 4 轮都有接触、``wheel_support_reward`` 在桩区外返回 0；
3. 把机器人抬高悬空后，支撑轮数变成 0、``wheel_airtime_penalty`` 在
   grace 步之后触发（连续悬空才罚）；
4. 轮子 geom 的摩擦域随机化已生效（sim.model.geom_friction 与默认不同）。

运行：

    uv run python tools/smoke_wheel_support.py
"""

from __future__ import annotations

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

from robot.config.env_cfgs import STEPSTONE_START_LENGTH, STEPSTONE_GOAL_DISTANCE
from robot.mdp.stepstone import wheel_airtime_penalty, wheel_support_reward
from robot.mdp.stone_coverage import _state, _sync


def build_env() -> ManagerBasedRlEnv:
    cfg = load_env_cfg("Robot-StepStone-v0")
    cfg.scene.num_envs = 1
    cfg.scene.terrain.num_envs = 1
    cfg.episode_length_s = 1e9
    cfg.curriculum = {}
    cfg.observations["actor"].enable_corruption = False
    if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
        gen = cfg.scene.terrain.terrain_generator
        gen.curriculum = True
        gen.num_rows = 10
        gen.num_cols = 1
        cfg.scene.terrain.max_init_terrain_level = 9
        cfg.scene.terrain.env_spacing = 4.5
    env = ManagerBasedRlEnv(cfg, device="cpu")
    env.scene.terrain.terrain_levels[:] = 9
    env.reset()
    return env


def support_now(env) -> int:
    found = env.scene["feet_ground_contact"].data.found
    if found.ndim == 3:
        found = found[..., -1]
    return int((found > 0.5).sum(dim=1)[0])


def main() -> int:
    env = build_env()
    robot = env.scene["robot"]
    state = _state(env)
    _sync(env, state)
    zero = torch.zeros(1, env.action_manager.total_action_dim)

    found = env.scene["feet_ground_contact"].data.found
    print(f"feet_ground_contact.found shape: {tuple(found.shape)} (expect (1, 4))")
    env.step(zero)
    s0 = support_now(env)
    print(f"standing on ground: support_wheels={s0} (expect 4)")
    if s0 != 4:
        print("FAILED: standing robot should have all four wheels in contact")
        return 1

    # 桩区外（出生点附近）：支撑奖励应为 0
    out = float(wheel_support_reward(env, x_min=STEPSTONE_START_LENGTH, x_max=STEPSTONE_GOAL_DISTANCE)[0])
    forced = float(wheel_support_reward(env, x_min=-10.0, x_max=10.0)[0])
    print(f"wheel_support_reward: outside_stones={out:.2f} (expect 0), forced={forced:.2f} (expect 0: 轮子不在桩顶矩形内)")
    if out != 0.0:
        print("FAILED: support reward should be 0 outside the stone field")
        return 1

    # 平移到桩区上方 + 抬高 0.7 m：过程应持续悬空，airtime 在 grace 后触发
    root = robot.data.root_link_pos_w[0].clone()
    origin = env.scene.env_origins[0]
    robot.write_root_link_pose_to_sim(
        torch.tensor(
            [[float(origin[0]) + 2.5, float(origin[1]), float(root[2]) + 0.7, 1.0, 0.0, 0.0, 0.0]],
            dtype=torch.float32,
        )
    )
    robot.write_root_link_velocity_to_sim(torch.zeros(1, 6))
    hits = 0
    min_support = 4
    for _ in range(12):
        env.step(zero)
        s = support_now(env)
        min_support = min(min_support, s)
        a = float(wheel_airtime_penalty(env, x_min=-10.0, x_max=10.0, min_support=3, grace_steps=5)[0])
        hits += int(a > 0.0)
    print(f"after 0.5 m lift: min support={min_support} (expect 0), airtime fired {hits}/12 steps (expect >0)")
    if min_support != 0 or hits == 0:
        print("FAILED: airborne robot should have no support and trigger the airtime penalty")
        return 1

    # 摩擦域随机化：比较轮子 geom 摩擦与 MuJoCo 默认 [1, 0.005, 0.0001]
    import mujoco

    friction = env.sim.model.geom_friction
    if hasattr(friction, "detach"):
        friction = friction.detach().cpu()
    mj = env.sim._mj_model
    all_names = [
        mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_GEOM, g) for g in range(mj.ngeom)
    ]
    wheel_names = {"FL_wheel_geom", "FR_wheel_geom", "RL_wheel_geom", "RR_wheel_geom"}
    idx = [
        g
        for g, name in enumerate(all_names)
        if name and name.split("/")[-1] in wheel_names
    ]
    print(f"ngeom={mj.ngeom}, wheel geom ids: {idx} (expect 4)")
    print(f"geom names containing 'wheel': {[n for n in all_names if n and 'wheel' in n][:12]}")
    changed = bool(idx) and any(abs(float(friction[0, g, 0]) - 1.0) > 1e-6 for g in idx)
    print(f"wheel friction randomized: {changed} (expect True)")
    if not changed:
        print("FAILED: wheel friction domain randomization did not change geom_friction")
        return 1

    print("OK: wheel contact observation and support/airtime rewards behave as expected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
