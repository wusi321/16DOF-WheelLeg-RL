"""统计策略在桩道上的四轮支撑情况（每步几个轮子着地/着桩）。

回答的核心问题：后腿轮子是不是"搭便车"悬空，靠前轮或碰撞带过去？

运行：

    uv run python tools/check_support_stats.py --checkpoint-file <ckpt> --level 6
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from dataclasses import asdict

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

from robot.config.env_cfgs import STEPSTONE_START_LENGTH, STEPSTONE_GOAL_DISTANCE
from robot.mdp.stepstone import wheel_support_reward


def build_env(levels: list[int]) -> ManagerBasedRlEnv:
    env_cfg = load_env_cfg("Robot-StepStone-v0", play=True)
    num_envs = len(levels)
    env_cfg.scene.num_envs = num_envs
    if env_cfg.scene.terrain is not None:
        env_cfg.scene.terrain.num_envs = num_envs
        gen = env_cfg.scene.terrain.terrain_generator
        if gen is not None:
            gen.curriculum = True
            gen.num_rows = 10
            gen.num_cols = 1
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
    terrain = env.scene.terrain
    max_level = int(terrain.terrain_origins.shape[0]) - 1
    lv = torch.tensor([min(v, max_level) for v in levels], dtype=torch.long)
    terrain.terrain_levels[:] = lv.to(terrain.terrain_levels.device)
    terrain.terrain_types[:] = 0
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
    env.reset()
    return env


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-file", required=True)
    ap.add_argument("--level", type=int, default=6)
    ap.add_argument("--steps", type=int, default=900)
    args = ap.parse_args()

    env = build_env([args.level])
    env_wrapped = RslRlVecEnvWrapper(env)
    agent_cfg = load_rl_cfg("Robot-StepStone-v0")
    runner = MjlabOnPolicyRunner(env_wrapped, asdict(agent_cfg), device="cpu")
    runner.load(args.checkpoint_file, load_cfg={"actor": True}, strict=True, map_location="cpu")
    policy = runner.get_inference_policy(device="cpu")

    # 关节顺序 FL, FR, RL, RR
    support_hist = [0, 0, 0, 0, 0]
    strict_hist = [0, 0, 0, 0, 0]
    front_hist = [0, 0, 0]   # 前两轮着地个数
    hind_hist = [0, 0, 0]    # 后两轮着地个数
    in_zone = 0
    hind_bad = 0
    obs = env_wrapped.reset()
    for _ in range(args.steps):
        obs_t = obs[0] if isinstance(obs, tuple) else obs
        with torch.inference_mode():
            act = policy(obs_t.detach())
        obs, _, _, _ = env_wrapped.step(act)
        found = env.scene["feet_ground_contact"].data.found
        if found.ndim == 3:
            found = found[..., -1]
        f = (found > 0.5)[0]
        n = int(f.sum())
        support_hist[n] += 1
        strict = int(float(wheel_support_reward(env, x_min=-10.0, x_max=10.0)[0]))
        strict_hist[min(strict, 4)] += 1
        front_hist[int(f[0]) + int(f[1])] += 1
        hind_hist[int(f[2]) + int(f[3])] += 1
        origin = env.scene.env_origins[0]
        root = env.scene["robot"].data.root_link_pos_w[0]
        x_rel = float(root[0] - origin[0])
        if STEPSTONE_START_LENGTH < x_rel < STEPSTONE_GOAL_DISTANCE:
            in_zone += 1
            if not (bool(f[2]) or bool(f[3])):
                hind_bad += 1

    total = sum(support_hist)
    print(f"\n===== support stats: level {args.level}, {total} steps =====")
    for n, c in enumerate(support_hist):
        print(f"  {n} wheels grounded (contact sensor): {c:4d} ({100.0 * c / total:5.1f}%)")
    for n, c in enumerate(strict_hist):
        print(f"  {n} wheels ON stone top (strict): {c:4d} ({100.0 * c / total:5.1f}%)")
    print(f"  front grounded count: 0={front_hist[0]} 1={front_hist[1]} 2={front_hist[2]}")
    print(f"  hind  grounded count: 0={hind_hist[0]} 1={hind_hist[1]} 2={hind_hist[2]}")
    if in_zone:
        print(
            f"in stone field: {in_zone} steps, "
            f"NO hind wheel grounded {hind_bad} steps ({100.0 * hind_bad / in_zone:5.1f}%)"
        )
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
