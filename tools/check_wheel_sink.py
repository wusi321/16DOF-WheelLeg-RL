"""量化轮子在桩上的穿透/下沉深度。

对每一步，找每个轮子最近的一根桩（xy 距离 < 0.2 m 才算在桩上），计算
轮底（轮心 z - 轮半径）与桩顶 z 的差：负值 = 轮子陷进桩里。

运行：

    uv run python tools/check_wheel_sink.py --checkpoint-file <ckpt> --level 6
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
from robot import naming


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
    terrain.terrain_levels[:] = lv
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
    state = sc._state(env)
    sc._sync(env, state)
    env_wrapped = RslRlVecEnvWrapper(env)
    agent_cfg = load_rl_cfg("Robot-StepStone-v0")
    runner = MjlabOnPolicyRunner(env_wrapped, asdict(agent_cfg), device="cpu")
    runner.load(args.checkpoint_file, load_cfg={"actor": True}, strict=True, map_location="cpu")
    policy = runner.get_inference_policy(device="cpu")

    robot = env.scene["robot"]
    obs = env_wrapped.reset()
    sinks: list[float] = []
    wheel_stones = 0
    for _ in range(args.steps):
        obs_t = obs[0] if isinstance(obs, tuple) else obs
        with torch.inference_mode():
            act = policy(obs_t.detach())
        obs, _, _, _ = env_wrapped.step(act)
        sc._sync(env, state)
        origin = env.scene.env_origins[0]
        wpos = robot.data.body_link_pos_w[0, state.wheel_ids] - origin  # (4,3) 相对 env
        sxy = state.stones[0, :, :2]
        sz = state.stones[0, :, 4]  # (S,5) = x, y, half_x, half_y, top
        shx = state.stones[0, :, 2]
        shy = state.stones[0, :, 3]
        if sxy.numel() == 0:
            continue
        for w in range(4):
            d = (wpos[w, None, :2] - sxy).norm(dim=-1)
            j = int(d.argmin())
            # 只在轮心落在桩顶矩形内时统计，排除"轮子站在桩边缘/滑落"的样本
            inside = (
                abs(float(wpos[w, 0] - sxy[j, 0])) <= float(shx[j])
                and abs(float(wpos[w, 1] - sxy[j, 1])) <= float(shy[j])
            )
            if inside:
                wheel_stones += 1
                wheel_bottom = float(wpos[w, 2]) - naming.WHEEL_RADIUS
                stone_top = float(sz[j])
                sinks.append(stone_top - wheel_bottom)  # 正 = 穿透/下沉

    if not sinks:
        print("no wheel-on-stone samples")
        return 1
    t = torch.tensor(sinks)
    print(f"\n===== wheel sink depth: level {args.level}, {len(sinks)} wheel-on-stone samples =====")
    print(f"  mean={t.mean()*1000:+.1f} mm  std={t.std()*1000:.1f} mm  min={t.min()*1000:+.1f}  max={t.max()*1000:+.1f} mm")
    for q in (0.5, 0.9, 0.99):
        print(f"  p{int(q*100)}={torch.quantile(t, q)*1000:+.1f} mm")
    print("  (正值 = 轮底低于桩顶 = 穿透下沉)")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
