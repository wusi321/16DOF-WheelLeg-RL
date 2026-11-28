"""诊断 StepStone 策略的步态：它到底踩了哪些桩、机身横在哪。

用法:
    uv run python tools/eval_stepstone_gait.py --checkpoint-file <ckpt> [--levels 0,3,6,9] [--steps 1200]

输出:
- 机身相对赛道中线 (y_rel = 0) 的分布：均值/std/直方图；
- 每根桩是否被踩、被哪个轮子踩（左列 y<0.8 / 右列 y>0.8）；
- 踩桩时该轮的横向偏移统计（判断"差一点没踩上"还是"根本没走那条线"）。
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

from robot.mdp import stone_coverage as sc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-file", required=True)
    parser.add_argument("--levels", default="0,3,6,9")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--task", default="Robot-StepStone-v0")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    ckpt = pathlib.Path(args.checkpoint_file)
    if not ckpt.exists():
        print(f"checkpoint not found: {ckpt}")
        return 1
    levels = [int(v) for v in args.levels.split(",")]
    num_envs = len(levels)

    env_cfg = load_env_cfg(args.task, play=True)
    env_cfg.scene.num_envs = num_envs
    if env_cfg.scene.terrain is not None:
        env_cfg.scene.terrain.num_envs = num_envs
        gen = env_cfg.scene.terrain.terrain_generator
        if gen is not None:
            # 必须 curriculum=True：难度沿行递增(row=level)，row 9 才是真
            # level 9。curriculum=False 时 difficulty 是 difficulty_range
            # 内的随机值，评估难度与 level 无关（mjlab/src/mjlab/terrains/
            # terrain_generator.py:229-230）。
            gen.curriculum = True
            gen.num_rows = 10
            gen.num_cols = 1

    env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
    terrain = env.scene.terrain
    assert terrain is not None and terrain.terrain_origins is not None
    max_level = int(terrain.terrain_origins.shape[0]) - 1
    lv = torch.tensor([min(v, max_level) for v in levels], dtype=torch.long)
    terrain.terrain_levels[:] = lv.to(terrain.terrain_levels.device)
    terrain.terrain_types[:] = 0
    terrain.env_origins[:] = terrain.terrain_origins[
        terrain.terrain_levels, terrain.terrain_types
    ]
    env.reset()

    agent_cfg = load_rl_cfg(args.task)
    env_wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = MjlabOnPolicyRunner(env_wrapped, asdict(agent_cfg), device=args.device)
    runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=args.device)
    policy = runner.get_inference_policy(device=args.device)

    state = sc._state(env)
    robot = env.scene["robot"]
    origins = env.scene.env_origins

    # 每环境的桩按左右列统计
    n_stones = state.stones.shape[1]
    lane_of = torch.zeros(num_envs, n_stones, dtype=torch.long)
    for i in range(num_envs):
        for s in range(n_stones):
            lane_of[i, s] = 1 if state.stones[i, s, 1] > 0.15 else 0  # 相对 y>0.15 → 右列

    visited = torch.zeros(num_envs, n_stones, dtype=torch.bool)
    y_rel_hist: list[float] = []          # 机身相对中线（相对出生点 y）
    miss_gap: list[float] = []            # 被踩轮到桩心的最近横向距（诊断容差）
    wheel_lane_touch = torch.zeros(num_envs, 2)  # 左列/右列累计被踩次数

    obs = env_wrapped.reset()
    for t in range(args.steps):
        obs_t = obs[0] if isinstance(obs, tuple) else obs
        with torch.inference_mode():
            act = policy(obs_t.detach())
        obs, _, _, _ = env_wrapped.step(act)
        sc._sync(env, state)
        visited |= state.visited

        y_rel = (robot.data.root_link_pos_w[:, 1] - origins[:, 1]).tolist()
        y_rel_hist.extend(y_rel)

        # 当前每个轮子离最近桩的横向距离
        wpos = robot.data.body_link_pos_w[:, state.wheel_ids] - origins[:, None, :]
        for i in range(num_envs):
            if not state.valid[i].any():
                continue
            sxy = state.stones[i, :, :2]  # (S,2)
            for w in range(4):
                d = (wpos[i, w, None, :2] - sxy).norm(dim=-1)
                d = d[state.valid[i]]
                if d.numel():
                    miss_gap.append(float(d.min()))

        for s in range(n_stones):
            if state.visited[0 if False else slice(None), s].any():
                pass
        for i in range(num_envs):
            for s in range(n_stones):
                if state.valid[i, s] and state.visited[i, s]:
                    wheel_lane_touch[i, lane_of[i, s]] += 1

    print(f"\n===== gait diagnostics ({ckpt.name}, steps={args.steps}) =====")
    for i, lv_i in enumerate(levels):
        cov = float((state.visited[i] & state.valid[i]).sum()) / max(1, int(state.valid[i].sum()))
        n_left = int(lane_of[i][state.valid[i]] == 0).sum() if False else int((lane_of[i] == 0)[state.valid[i]].sum()) if False else sum(1 for s in range(n_stones) if state.valid[i, s] and lane_of[i, s] == 0)
        n_right = sum(1 for s in range(n_stones) if state.valid[i, s] and lane_of[i, s] == 1)
        v_left = sum(1 for s in range(n_stones) if state.valid[i, s] and state.visited[i, s] and lane_of[i, s] == 0)
        v_right = sum(1 for s in range(n_stones) if state.valid[i, s] and state.visited[i, s] and lane_of[i, s] == 1)
        print(f"level {lv_i}: stones L={n_left} R={n_right} | visited L={v_left} R={v_right} | coverage={cov:.2f}")

    ys = torch.tensor(y_rel_hist)
    print(f"\nbody y_rel: mean={ys.mean():+.3f} std={ys.std():.3f}")
    hist = torch.histc(ys, bins=16, min=-0.6, max=0.6)
    edges = [-0.6 + i * 1.2 / 16 for i in range(17)]
    print("y_rel histogram (relative to track centreline 0):")
    for i in range(16):
        bar = "#" * int(60 * hist[i] / max(1, hist.max()))
        print(f"  [{edges[i]:+.2f},{edges[i+1]:+.2f}) {hist[i]:6.0f} {bar}")

    gaps = torch.tensor(miss_gap)
    print(f"\nwheel->nearest-stone 2D dist: mean={gaps.mean():.3f}  p50={gaps.median():.3f}  "
          f"p10={torch.quantile(gaps, 0.1):.3f}")
    print("(<=0.15 会记为踩到: 桩半宽 0.10 + XY_TOL 0.05)")

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
