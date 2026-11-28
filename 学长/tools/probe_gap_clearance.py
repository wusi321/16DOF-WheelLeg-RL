"""统计轮子经过两根桩之间缝隙时的离地余量（相对桩顶高度）。

对每一步的每个轮子：
- 只看桩区内的轮子（相对出生点 x 在 0.8~6.8 m）；
- 只统计轮心**不在任何桩顶矩形内**（即正处在缝隙上方）的样本；
- 记录轮心 z 与最近桩顶 z 的差（负值 = 轮心已低于桩顶，正 = 高于桩顶）。

分前轮 / 后轮统计，输出分位数和"轮心低于桩顶"的占比。

运行：

    uv run python tools/probe_gap_clearance.py --checkpoint-file <ckpt> --level 6 --steps 900
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


def build_env(level: int) -> ManagerBasedRlEnv:
    env_cfg = load_env_cfg("Robot-StepStone-v0", play=True)
    env_cfg.scene.num_envs = 1
    if env_cfg.scene.terrain is not None:
        env_cfg.scene.terrain.num_envs = 1
        gen = env_cfg.scene.terrain.terrain_generator
        if gen is not None:
            gen.curriculum = True
            gen.num_rows = 10
            gen.num_cols = 1
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
    terrain = env.scene.terrain
    max_level = int(terrain.terrain_origins.shape[0]) - 1
    lv = torch.tensor([min(level, max_level)], dtype=torch.long)
    terrain.terrain_levels[:] = lv
    terrain.terrain_types[:] = 0
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
    env.reset()
    return env


def summarize(name: str, vals: list[float]) -> None:
    if not vals:
        print(f"  {name}: no samples")
        return
    t = torch.tensor(vals)
    print(
        f"  {name}: n={len(vals)}  mean={t.mean()*100:+.1f} cm  "
        f"p10={torch.quantile(t, 0.10)*100:+.1f}  p50={torch.quantile(t, 0.50)*100:+.1f}  "
        f"p90={torch.quantile(t, 0.90)*100:+.1f} cm   below_top={float((t < 0).float().mean())*100:.1f}%  "
        f"below_top+2cm={float((t < -0.02).float().mean())*100:.1f}%"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-file", required=True)
    ap.add_argument("--level", type=int, default=6)
    ap.add_argument("--steps", type=int, default=900)
    args = ap.parse_args()

    env = build_env(args.level)
    state = sc._state(env)
    sc._sync(env, state)
    env_wrapped = RslRlVecEnvWrapper(env)
    agent_cfg = load_rl_cfg("Robot-StepStone-v0")
    runner = MjlabOnPolicyRunner(env_wrapped, asdict(agent_cfg), device="cpu")
    runner.load(args.checkpoint_file, load_cfg={"actor": True}, strict=True, map_location="cpu")
    policy = runner.get_inference_policy(device="cpu")

    robot = env.scene["robot"]
    obs = env_wrapped.reset()
    front: list[float] = []
    hind: list[float] = []
    for _ in range(args.steps):
        obs_t = obs[0] if isinstance(obs, tuple) else obs
        with torch.inference_mode():
            act = policy(obs_t.detach())
        obs, _, _, _ = env_wrapped.step(act)
        sc._sync(env, state)
        origin = env.scene.env_origins[0]
        wpos = robot.data.body_link_pos_w[0, state.wheel_ids] - origin  # (4,3)
        sxy = state.stones[0, :, :2]
        sz = state.stones[0, :, 4]
        shx = state.stones[0, :, 2]
        shy = state.stones[0, :, 3]
        if sxy.numel() == 0:
            continue
        x = float(wpos[0, 0])
        if not (0.8 <= x <= 6.8):
            continue
        for w in range(4):
            px, py = float(wpos[w, 0]), float(wpos[w, 1])
            if abs(py) > 0.9:  # 出赛道宽度
                continue
            # 在任一桩顶矩形内 -> 这是"在桩上"，跳过
            inside = (
                (torch.abs(sxy[:, 0] - px) <= shx) & (torch.abs(sxy[:, 1] - py) <= shy)
            ).any().item()
            if inside:
                continue
            # 缝隙上方：找最近桩的顶面高度
            d2 = (sxy[:, 0] - px) ** 2 + (sxy[:, 1] - py) ** 2
            j = int(d2.argmin())
            dz = float(wpos[w, 2]) - float(sz[j])
            if w < 2:
                front.append(dz)
            else:
                hind.append(dz)

    print(f"\n===== gap clearance: level {args.level}, {args.steps} steps =====")
    print("  dz = wheel center z - nearest stone top z  (negative = wheel center below stone top)")
    summarize("front wheels (wheels over gaps)", front)
    summarize("hind wheels  (wheels over gaps)", hind)
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
