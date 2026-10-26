"""按指定难度回放 StepStone 策略。

`mjlab/scripts/play.py` 只会把课程关掉、所有环境放在 level 0，看不到高难度。
这个脚本把 N 个环境**确定性地**分配到 level 0..N-1（每个环境一个难度），
再加载 checkpoint 跑 viewer。

    # 10 个环境，每个难度一个
    uv run python tools/play_stepstone_levels.py \
        --checkpoint-file logs/rsl_rl/robot_stepstone/<run>/model_500.pt

    # 只看第 9 级（最高难度）
    uv run python tools/play_stepstone_levels.py \
        --checkpoint-file .../model_500.pt --levels 9

    # 120 个环境铺满 12 列 x 10 行，但每个等级仍只有 1 个 patch 位置
    uv run python tools/play_stepstone_levels.py --checkpoint-file ... --num-envs 30
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.viewer import ViserPlayViewer


def parse_levels(spec: str, num_envs: int, max_level: int) -> torch.Tensor:
    """``spec`` 可以是单个数字（全部用该等级）或 "0,3,9" 这样的列表。"""
    if "," in spec:
        levels = [int(v) for v in spec.split(",")]
    else:
        levels = [int(spec)]
    out = [levels[i % len(levels)] for i in range(num_envs)]
    return torch.tensor([min(v, max_level) for v in out], dtype=torch.long)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-file", required=True)
    parser.add_argument("--num-envs", type=int, default=10, help="环境数（默认 10 = 每个难度一个）")
    parser.add_argument("--levels", default="", help="难度列表，如 9 或 0,3,6,9；默认 0..N-1")
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
        if env_cfg.scene.terrain.terrain_generator is not None:
            env_cfg.scene.terrain.terrain_generator.curriculum = False

    env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)

    terrain = env.scene.terrain
    assert terrain is not None and terrain.terrain_origins is not None
    max_level = int(terrain.terrain_origins.shape[0]) - 1
    levels = (
        parse_levels(args.levels, args.num_envs, max_level)
        if args.levels
        else torch.arange(args.num_envs) % (max_level + 1)
    )
    # 先把每个环境钉到指定难度的 patch 上，再 reset（reset 会按 env_origins 摆机器人）。
    terrain.terrain_levels[:] = levels.to(terrain.terrain_levels.device)
    terrain.terrain_types[:] = 0
    terrain.env_origins[:] = terrain.terrain_origins[
        terrain.terrain_levels, terrain.terrain_types
    ]
    env.reset()
    print("levels:", sorted(set(int(v) for v in levels.tolist())))

    agent_cfg = load_rl_cfg(args.task)
    env_wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = MjlabOnPolicyRunner(env_wrapped, _asdict(agent_cfg), device=args.device)
    runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=args.device)
    policy = runner.get_inference_policy(device=args.device)

    ViserPlayViewer(env_wrapped, policy).run()
    env.close()
    return 0


def _asdict(cfg) -> dict:
    from dataclasses import asdict

    return asdict(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
