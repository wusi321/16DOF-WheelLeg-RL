"""Verify the "touch every stone top" rule on a real StepStone environment.

Checks that

1. stones are recovered from the compiled scene and assigned to the right env;
2. standing on the start platform touches nothing;
3. a wheel on a stone top registers a new touch worth ``+3``;
4. the same stone does not pay twice;
5. visiting every stone drives coverage to 1.0 and pays the completion bonus;
6. two or more wheels on the pit floor cost ``-2`` per step.

Runs on CPU (first run compiles mujoco-warp kernels):

    uv run python tools/check_stepstone_coverage.py
"""

from __future__ import annotations

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

from robot.mdp.stone_coverage import (
    _state,
    _sync,
    all_stones_bonus,
    pit_contact_penalty,
    stone_coverage,
    stone_touch_reward,
)

WHEEL_RADIUS = 0.086
PIT_DEPTH = 0.6


def build_env() -> ManagerBasedRlEnv:
    cfg = load_env_cfg("Robot-StepStone-v0")
    cfg.scene.num_envs = 1
    cfg.scene.terrain.num_envs = 1
    if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.num_rows = 2
        cfg.scene.terrain.terrain_generator.num_cols = 2
    env = ManagerBasedRlEnv(cfg, device="cpu")
    env.reset()
    return env


def place_wheel(robot, rel, stone_world, wheel_index: int = 0) -> None:
    """Teleport so that ``wheel_index`` sits exactly on the given stone top."""
    sx, sy, stop = stone_world
    rx, ry, rz = (float(rel[wheel_index][k]) for k in range(3))
    root = torch.tensor(
        [[sx - rx, sy - ry, stop + WHEEL_RADIUS - rz, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32
    )
    robot.write_root_link_pose_to_sim(root)
    robot.write_root_link_velocity_to_sim(torch.zeros(1, 6))


def main() -> int:
    env = build_env()
    robot = env.scene["robot"]
    state = _state(env)
    counts = state.extras["counts"].tolist()
    total = int(counts[0])
    print(f"stones per env: {counts} (total {total})")
    if total == 0:
        print("FAILED: no stones recovered from the scene")
        return 1

    origin = env.scene.env_origins[0]
    zero = torch.zeros(1, env.action_manager.total_action_dim)

    def stone_world(i: int) -> tuple[float, float, float]:
        x, y, _hx, _hy, top = (float(v) for v in state.stones[0, i])
        return float(origin[0]) + x, float(origin[1]) + y, float(origin[2]) + top

    env.step(zero)
    root = robot.data.root_link_pos_w[0]
    wheels = robot.data.body_link_pos_w[0, state.wheel_ids]
    rel = (wheels - root).detach().cpu()

    # --- 2/3: platform pays nothing, a stone pays 3 -------------------------
    env.episode_length_buf += 1  # emulate a new policy step (the caches key on it)
    spawn = float(_sync(env, state).sum())
    print(f"on start platform: newly touched = {spawn:.0f} (expect 0)")
    if spawn != 0:
        print("FAILED: standing on the platform counted as a stone touch")
        return 1

    touch_w = float(env.cfg.rewards["stone_touch"].weight)
    all_w = float(env.cfg.rewards["all_stones"].weight)
    place_wheel(robot, rel, stone_world(0))
    env.sim.forward()
    env.episode_length_buf += 1
    reward = float(stone_touch_reward(env, scale=touch_w)[0])
    cov = float(stone_coverage(env)[0])
    print(f"wheel on stone 0: reward = {reward:.1f} (expect {touch_w:.1f}), coverage = {cov:.3f}")
    if abs(reward - touch_w) > 1e-6:
        print("FAILED: touching a stone did not pay the configured weight")
        return 1

    # --- 4: no double payment ----------------------------------------------
    env.episode_length_buf += 1  # pretend a new policy step, same pose
    again = float(stone_touch_reward(env, scale=touch_w)[0])
    print(f"same stone, next step: reward = {again:.1f} (expect 0.0)")
    if again != 0.0:
        print("FAILED: the same stone paid twice")
        return 1

    # --- 5: visit everything ------------------------------------------------
    # Teleport rather than step: one wheel on a stone is not a stable stance, and a
    # fall would reset the episode (and therefore the coverage mask).
    for i in range(1, total):
        place_wheel(robot, rel, stone_world(i), wheel_index=i % 4)
        env.sim.forward()
        env.episode_length_buf += 1
        stone_touch_reward(env, scale=touch_w)

    cov = float(stone_coverage(env)[0])
    bonus = float(all_stones_bonus(env, scale=all_w)[0])
    print(f"after visiting all stones: coverage = {cov:.2f} (expect 1.00), "
          f"all-stones bonus = {bonus:.1f} (expect {all_w:.1f})")
    if cov < 0.999 or bonus <= 0.0:
        print("FAILED: full coverage was not detected")
        return 1

    # --- 6: half the wheels on the pit floor --------------------------------
    pit_root_z = -PIT_DEPTH + WHEEL_RADIUS - float(rel[0][2])
    robot.write_root_link_pose_to_sim(
        torch.tensor([[float(origin[0]) + 2.0, 0.0, pit_root_z, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    )
    robot.write_root_link_velocity_to_sim(torch.zeros(1, 6))
    env.sim.forward()
    pit_w = float(env.cfg.rewards["pit_contact"].weight)
    penalty = float(pit_contact_penalty(env, scale=pit_w)[0])
    print(f"wheels on the pit floor: penalty = {penalty:.1f} (expect {pit_w:.1f})")
    if penalty >= 0.0:
        print("FAILED: pit-floor contact was not penalised")
        return 1

    env.close()
    print("OK: stone coverage rule works end to end")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
