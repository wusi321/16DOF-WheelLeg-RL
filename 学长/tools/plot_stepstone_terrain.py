"""把梅花桩地形按难度逐级画成俯视图，用来核对间距/净空，不用跑训练。

    uv run python tools/plot_stepstone_terrain.py            # 输出 docs/images/terrain_levels.png
    uv run python tools/plot_stepstone_terrain.py --out x.png

每个子图是一个难度等级的 patch（8 m × 1.6 m），绿色方块是桩（200×200 mm），
浅绿色是两端平台（各 0.8 m），标注了同侧相邻净空和首桩净空。
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from robot.config.env_cfgs import STEPSTONE_NUM_LEVELS, STEPSTONE_TERRAIN_SIZE
from robot.mdp.stone_coverage import extract_stones

STONE = 0.2
START = 0.8
END = 0.8


def build_stones() -> tuple[list[tuple[float, float]], np.ndarray, float]:
    cfg = load_env_cfg("Robot-StepStone-v0")
    cfg.scene.num_envs = 2
    cfg.scene.terrain.num_envs = 2
    env = ManagerBasedRlEnv(cfg, device="cpu")
    env.reset()
    terrain = env.scene.terrain
    origins = terrain.terrain_origins.detach().cpu().numpy()
    stones = extract_stones(env.sim._mj_model, half_xy=(STONE / 2, STONE / 2))
    env.close()
    return stones, origins, float(STEPSTONE_TERRAIN_SIZE[0])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="docs/images/terrain_levels.png")
    args = parser.parse_args()

    stones, origins, sx = build_stones()
    levels = STEPSTONE_NUM_LEVELS

    fig, axes = plt.subplots(
        levels, 1, figsize=(14, 1.45 * levels), sharex=True, constrained_layout=True
    )
    for lv, ax in enumerate(axes):
        ox, oy, _ = origins[lv, 0]
        # terrain_origins 存的是**出生点**（起点平台中心，patch x = 0.4），
        # 所以下面所有坐标都是「相对出生点」的：平台在 [-0.4, 0.4]。
        pts = [
            (float(p[0] - ox), float(p[1] - oy))
            for p in stones
            if -0.6 <= p[0] - ox <= sx - START / 2 and abs(p[1] - oy) <= 1.0
        ]
        goal = sx - END - START / 2  # 6.8 m
        # 平台：起点 [-0.4, 0.4]，终点 [6.8, 7.6]
        for x0, width in ((-START / 2, START), (goal, END)):
            ax.add_patch(patches.Rectangle((x0, -0.8), width, 1.6,
                                           facecolor="#bfe9c8", edgecolor="#7fbf8f", lw=0.6, zorder=1))
        for x, y in pts:
            ax.add_patch(patches.Rectangle((x - STONE / 2, y - STONE / 2), STONE, STONE,
                                           facecolor="#2e8b3d", edgecolor="k", lw=0.4, zorder=3))
        # 间距标注
        left = sorted(x for x, y in pts if y <= 0)
        right = sorted(x for x, y in pts if y > 0)
        gaps = [b - a for xs in (left, right) for a, b in zip(xs, xs[1:])]
        first = min((min(xs) - STONE / 2 - START / 2 for xs in (left, right) if xs), default=0.0)
        info = f"L{lv}  max_same_lane_gap={max(gaps) - STONE:.3f}m  first_gap={first:.3f}m  n={len(pts)}"
        ax.text(0.002, 0.94, info, transform=ax.transAxes, fontsize=8.5, va="top", family="monospace")
        ax.set_xlim(-0.5, goal + END + 0.1)
        ax.set_ylim(-0.85, 0.85)
        ax.set_yticks([])
        ax.set_ylabel(f"L{lv}", rotation=0, labelpad=14, fontsize=9)
        ax.axvline(goal, color="#e0a33a", lw=0.8, ls="--", zorder=2)
        ax.grid(True, axis="x", lw=0.3, alpha=0.5)
    axes[-1].set_xlabel("x [m]  (机器人从 0 出发，进入 7.2 m 后的终点平台即算通过)")
    fig.suptitle(
        "StepStone terrain by difficulty (top view, 8.0 m x 1.6 m patch)", fontsize=12
    )
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
