"""统计轮子落桩精度：轮心离桩中心多远、有没有踩在桩顶边缘。

回答的问题：轮子是不是常常落在桩顶"前边缘/侧边缘"，只搭上一小半？

运行：

    uv run python tools/check_landing_precision.py --checkpoint-file <ckpt> --level 6
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

from robot.mdp.stone_coverage import _state, _sync
from robot.naming import WHEEL_RADIUS

WHEEL_NAMES = ("FL", "FR", "RL", "RR")
EDGE_TOL = 0.06  # 轮心在桩顶矩形外 6cm 内 = 踩边
ON_TOP_TOL = 0.06


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

    state = _state(env)
    robot = env.scene["robot"]

    samples: list[list[tuple[float, float]]] = [[], [], [], []]  # 每轮 (dx, dy)
    edge_samples = [0, 0, 0, 0]  # 轮心在桩顶矩形外、但仍压在桩顶边缘上
    edge_hits = [0, 0, 0, 0]

    obs = env_wrapped.reset()
    for _ in range(args.steps):
        obs_t = obs[0] if isinstance(obs, tuple) else obs
        with torch.inference_mode():
            act = policy(obs_t.detach())
        obs, _, _, _ = env_wrapped.step(act)
        _sync(env, state)

        origins = env.scene.env_origins[0]
        pos = robot.data.body_link_pos_w[0, state.wheel_ids] - origins
        stones = state.stones[0]
        valid = state.valid[0]
        sx, sy = stones[:, 0], stones[:, 1]
        shx, shy = stones[:, 2], stones[:, 3]
        top = stones[:, 4]
        for w in range(4):
            x, y = float(pos[w, 0]), float(pos[w, 1])
            bottom = float(pos[w, 2]) - WHEEL_RADIUS
            dx = (x - sx).abs()
            dy = (y - sy).abs()
            inside = (dx <= shx) & (dy <= shy) & valid
            on_top = inside & ((bottom - top).abs() <= ON_TOP_TOL)
            if bool(on_top.any()):
                idx = int(on_top.nonzero()[0])
                samples[w].append((float(dx[idx]), float(dy[idx])))
                continue
            # 踩边：轮心不在矩形内，但在边缘 6cm 内且有桩顶边缘低于轮子
            near = (dx <= shx + EDGE_TOL) & (dy <= shy + EDGE_TOL) & valid & ~inside
            if bool(near.any()):
                idx = int(near.nonzero()[0])
                edge_samples[w] += 1
                if bottom < float(top[idx]) + 0.02:
                    edge_hits[w] += 1

    print(f"\n===== landing precision: level {args.level}, {args.steps} steps =====")
    for w in range(4):
        s = samples[w]
        if not s:
            print(f"  {WHEEL_NAMES[w]}: no clean landings")
            continue
        t = torch.tensor(s)
        dx = t[:, 0]
        dy = t[:, 1]
        dist = (dx**2 + dy**2).sqrt()
        p = lambda v, q: float(torch.quantile(v, q)) * 100
        edge_ratio = float((dist > 0.08).float().mean()) * 100
        print(
            f"  {WHEEL_NAMES[w]}: n={len(s):4d}  |dx| p50={p(dx,0.5):4.1f} p90={p(dx,0.9):4.1f} cm"
            f"  |dy| p50={p(dy,0.5):4.1f} p90={p(dy,0.9):4.1f} cm"
            f"  dist p50={p(dist,0.5):4.1f} p90={p(dist,0.9):4.1f} cm"
            f"  near-edge(>8cm)={edge_ratio:4.1f}%"
        )
        if edge_samples[w]:
            print(
                f"      overlapping edge: {edge_samples[w]:3d} steps"
                f"  (wheel bottom below stone top on {edge_hits[w]} of them)"
            )

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
