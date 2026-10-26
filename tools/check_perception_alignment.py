"""Check that the perception encoder reproduces the training observation exactly.

This is the phase-1 golden test (``docs/perception/05_roadmap.md``). It builds a real
mjlab environment, steps it with random actions, and asserts that

    encode(SimTerrainSource(sensor).read())

equals the clipped ``safe_height_scan`` observation the policy is trained on.

Needs mjlab + mujoco-warp (CPU backend is fine, first run compiles kernels):

    uv run python tools/check_perception_alignment.py
    uv run python tools/check_perception_alignment.py --tasks Robot-StepStone-v0 --steps 20
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch

# ``perception`` lives at the repo root and is not an installed package.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from perception.encode import encode
from perception.frame import CLIP_MAX, CLIP_MIN
from perception.sources import SimTerrainSource

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from robot.mdp.rewards import safe_height_scan


def check(task_id: str, steps: int, num_envs: int, seed: int) -> float:
    cfg = load_env_cfg(task_id)
    cfg.scene.num_envs = num_envs
    cfg.scene.terrain.num_envs = num_envs
    if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.num_rows = 2
        cfg.scene.terrain.terrain_generator.num_cols = 2

    env = ManagerBasedRlEnv(cfg, device="cpu")
    torch.manual_seed(seed)
    env.reset()
    sensors = [SimTerrainSource(env.scene["height_scanner"], env_index=i) for i in range(num_envs)]
    n = env.action_manager.total_action_dim

    worst = 0.0
    for step in range(steps):
        actions = 0.5 * torch.randn(num_envs, n)
        env.step(actions)
        reference = safe_height_scan(env, "height_scanner").detach().cpu().numpy()
        reference = np.clip(reference, CLIP_MIN, CLIP_MAX)
        for i, source in enumerate(sensors):
            got = encode(source.read())
            diff = float(np.max(np.abs(got - reference[i])))
            worst = max(worst, diff)
    env.close()
    print(f"{task_id}: {steps} steps x {num_envs} envs, max |diff| = {worst:.3e}")
    return worst


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["Robot-Flat-v0", "Robot-StepStone-v0"],
        help="task ids to check",
    )
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()

    worst = 0.0
    for task in args.tasks:
        worst = max(worst, check(task, args.steps, args.num_envs, args.seed))

    if worst > args.tolerance:
        print(f"FAILED: max diff {worst:.3e} > {args.tolerance:.3e}")
        return 1
    print(f"OK: perception encoding matches the training observation (max diff {worst:.3e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
