"""按指定难度评估 StepStone 策略的**通过率**（到终点成功率），而不是踩桩覆盖率。

`tools/eval_stepstone_gait.py` 看的是"踩到了几根桩"；这个脚本看的是
"能不能走到终点"，以及失败原因分布。冲 7-9 级时这才是真正的验收指标。

用法：
    uv run python tools/eval_stepstone_success.py \
        --checkpoint-file logs/rsl_rl/robot_stepstone/<run>/model_999.pt \
        --levels 9 --num-envs 256 --max-steps 9000

输出：各终止原因占比、goal_reached 成功率、平均回合长度、平均石桩覆盖率。
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from collections import defaultdict
from dataclasses import asdict

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

from robot.mdp import stone_coverage as sc


def parse_levels(spec: str, num_envs: int, max_level: int) -> torch.Tensor:
    if "," in spec:
        levels = [int(v) for v in spec.split(",")]
    else:
        levels = [int(spec)]
    out = [levels[i % len(levels)] for i in range(num_envs)]
    return torch.tensor([min(v, max_level) for v in out], dtype=torch.long)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-file", required=True)
    parser.add_argument("--levels", default="9", help="难度，如 9 或 7,8,9")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--episodes", type=int, default=1, help="连续评估的回合数")
    parser.add_argument("--max-steps", type=int, default=9000, help="所有回合数超过该步数则截断")
    parser.add_argument("--task", default="Robot-StepStone-v0")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    ckpt = pathlib.Path(args.checkpoint_file)
    if not ckpt.exists():
        print(f"checkpoint not found: {ckpt}")
        return 1

    env_cfg = load_env_cfg(args.task, play=True)
    env_cfg.scene.num_envs = args.num_envs
    if env_cfg.scene.terrain is not None:
        env_cfg.scene.terrain.num_envs = args.num_envs
        gen = env_cfg.scene.terrain.terrain_generator
        if gen is not None:
            # 必须保持 curriculum=True：难度沿行递增，row 9 才是真 level 9。
            # 若设 curriculum=False，每个 patch 的 difficulty 是
            # difficulty_range 内的随机值（mjlab/src/mjlab/terrains/
            # terrain_generator.py:229-230），评估难度与 level 完全无关，
            # 之前"level 9 通过率 87.5%"就是这么来的假象。
            gen.curriculum = True
            # 固定 10 行（与训练一致），行号 = level。不能按 max_level+1
            # 缩行，否则 t=row/(num_rows-1) 会把低 level 也放大成最高难度。
            gen.num_rows = 10
            gen.num_cols = 1

    env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
    terrain = env.scene.terrain
    assert terrain is not None and terrain.terrain_origins is not None
    max_level = int(terrain.terrain_origins.shape[0]) - 1
    num_types = int(terrain.terrain_origins.shape[1])
    if args.num_envs > num_types:
        print(f"WARNING: num_envs={args.num_envs} > num_types={num_types}，patch 会重叠")
    levels = parse_levels(args.levels, args.num_envs, max_level)
    terrain.terrain_levels[:] = levels.to(terrain.terrain_levels.device)
    terrain.terrain_types[:] = (
        torch.arange(args.num_envs, device=terrain.terrain_types.device) % num_types
    )
    terrain.env_origins[:] = terrain.terrain_origins[
        terrain.terrain_levels, terrain.terrain_types
    ]
    env.reset()
    print(
        f"levels: {sorted(set(int(v) for v in levels.tolist()))}, envs: {args.num_envs}, "
        f"num_types: {num_types}"
    )

    agent_cfg = load_rl_cfg(args.task)
    env_wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = MjlabOnPolicyRunner(env_wrapped, asdict(agent_cfg), device=args.device)
    runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=args.device)
    policy = runner.get_inference_policy(device=args.device)

    term = env.termination_manager
    term_names = list(term.active_terms)
    num_envs = args.num_envs
    episodes_target = args.episodes * num_envs
    outcomes: dict[str, int] = defaultdict(int)
    ep_len_sum = 0
    coverage_sum = 0.0
    n_done = 0
    steps = 0
    # RslRlVecEnvWrapper.step 返回的 extras 实测不含 Episode_Metrics（空 dict），
    # 覆盖率不能从 extras 读。改为每步在 step 前用 stone_coverage 手动采样，
    # 终止时取终止前那一步的值（reset 后 visited 会清零）。
    state = sc._state(env)
    obs = env_wrapped.get_observations()

    while n_done < episodes_target and steps < args.max_steps:
        sc._sync(env, state)
        cov_now = (state.visited & state.valid).float().sum(dim=1) / (
            state.valid.float().sum(dim=1).clamp(min=1)
        )
        # 先记录 step 前的回合长度（reset 后会被清零）
        prev_len = env.unwrapped.episode_length_buf.clone()
        actions = policy(obs)
        obs, _, dones, extras = env_wrapped.step(actions)
        done_ids = torch.nonzero(dones, as_tuple=False).flatten()
        if done_ids.numel():
            for name in term_names:
                outcomes[name] += int(term.get_term(name)[done_ids].sum())
            ep_len_sum += int((prev_len[done_ids] + 1).sum())
            coverage_sum += float(cov_now[done_ids].sum())
            n_done += int(done_ids.numel())
        steps += 1

    total = int(n_done)
    print(f"\nfinished episodes: {total} (stop after {steps} steps)")
    print(f"mean episode length: {ep_len_sum / max(total, 1):.1f} steps")
    if total:
        print(f"mean stone coverage at episode end: {coverage_sum / total:.3f}")
    print("\ntermination breakdown (calls):")
    for name, count in sorted(outcomes.items(), key=lambda kv: -kv[1]):
        print(f"  {name:22s} {count:6d}  ({100.0 * count / max(total, 1):.1f}%)")
    success = outcomes.get("goal_reached", 0)
    print(f"\ngoal success rate: {100.0 * success / max(total, 1):.1f}%")

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
