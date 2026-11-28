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
    parser.add_argument(
        "--num-envs",
        type=int,
        default=None,
        help="环境数；默认=难度个数（--levels 有几个就开几个），不指定难度时=10（0~9 各一个）",
    )
    parser.add_argument("--levels", default="", help="难度列表，如 9 或 0,3,6,9；默认 0..N-1")
    parser.add_argument("--task", default="Robot-StepStone-v0")
    parser.add_argument(
        "--max-step",
        type=float,
        default=None,
        help="覆盖地形桩距上限（米）。p13 及更早的模型在 0.46 上训练，p29 之后在 0.44 上训练。",
    )
    parser.add_argument(
        "--legacy-obs330",
        action="store_true",
        help="去掉 wheel_offset（8 维）观测，回到 330 维 actor 观测，用于加载 p13 这类旧模型。",
    )
    parser.add_argument(
        "--legacy-noise",
        action="store_true",
        help="把单桩位置扰动改回 p13 时代的旧曲线（最高等级 3cm），用于加载旧模型。",
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    ckpt = pathlib.Path(args.checkpoint_file)
    if not ckpt.exists():
        print(f"checkpoint not found: {ckpt}")
        return 1

    levels_spec = [int(v) for v in args.levels.split(",")] if args.levels else None
    # 课程模式下每档难度只有一个桩阵（num_cols=1），同一难度开多个环境会
    # 全部叠在同一个出生点上（看上去"一堆模型"）。所以默认按难度个数开。
    num_envs = (
        args.num_envs
        if args.num_envs is not None
        else (len(levels_spec) if levels_spec else 10)
    )
    env_cfg = load_env_cfg(args.task, play=True)
    env_cfg.scene.num_envs = num_envs
    if env_cfg.scene.terrain is not None:
        env_cfg.scene.terrain.num_envs = num_envs
        gen = env_cfg.scene.terrain.terrain_generator
        if gen is not None:
            # 必须 curriculum=True：难度沿行递增（行号=level）。若设
            # curriculum=False，每个 patch 的 difficulty 是 difficulty_range
            # 内的随机值（mjlab/src/mjlab/terrains/terrain_generator.py:229-230），
            # 回放看到的难度跟 --levels 指定的完全无关。
            gen.curriculum = True
            gen.num_rows = 10
            gen.num_cols = 1
            if args.max_step is not None:
                gen.max_step = float(args.max_step)

    if args.legacy_obs330:
        for group in ("actor", "critic"):
            env_cfg.observations[group].terms.pop("wheel_offset", None)

    if args.legacy_noise:
        import numpy as np

        from robot.terrains import stepping_stones as _ss

        _ss._NOISE = np.array([0.00, 0.00, 0.01, 0.02, 0.025, 0.030], dtype=float)

    env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)

    terrain = env.scene.terrain
    assert terrain is not None and terrain.terrain_origins is not None
    max_level = int(terrain.terrain_origins.shape[0]) - 1
    levels = (
        parse_levels(args.levels, num_envs, max_level)
        if args.levels
        else torch.arange(num_envs) % (max_level + 1)
    )
    n_unique = len(set(int(v) for v in levels.tolist()))
    if num_envs > n_unique:
        print(
            f"WARNING: {num_envs} 个环境只占 {n_unique} 个难度，同一难度的环境会在"
            f"同一个桩阵上重叠。只看一个加 --num-envs 1；多看几档用 --levels 7,8,9。"
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
