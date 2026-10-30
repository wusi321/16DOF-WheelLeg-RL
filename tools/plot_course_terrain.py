"""把两条新赛道（沟壑 / 绕桩）按难度逐级画成俯视图，训练前先用它自查。

    uv run python tools/plot_course_terrain.py
    # -> docs/images/gapcourse_levels.png, docs/images/slalom_levels.png
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

LEVELS = 10
COLORS = {
    "slab": ("#2e8b3d", "slab"),
    "pillar": ("#e2731a", "pillar"),
    "waypoint": ("#3b8fd6", "waypoint"),
}


def _geoms(model) -> list[tuple[float, float, float, float]]:
    """terrain body 里的所有 geom：返回 (x, y, half_x, half_y)。"""
    terrain_body = None
    for b in range(model.nbody):
        name = model.body(b).name
        if name == "terrain":
            terrain_body = b
            break
    out = []
    for g in range(model.ngeom):
        if terrain_body is not None and int(model.geom_bodyid[g]) != terrain_body:
            continue
        s = model.geom_size[g]
        pos = model.geom_pos[g]
        out.append((float(pos[0]), float(pos[1]), float(s[0]), float(s[1])))
    return out


def render(task: str, out: pathlib.Path, title: str) -> None:
    cfg = load_env_cfg(task)
    cfg.scene.num_envs = LEVELS
    cfg.scene.terrain.num_envs = LEVELS
    cfg.scene.terrain.terrain_generator.num_rows = LEVELS
    cfg.scene.terrain.terrain_generator.num_cols = 1
    env = ManagerBasedRlEnv(cfg, device="cpu")
    env.reset()
    terrain = env.scene.terrain
    terrain.terrain_levels[:] = torch.arange(LEVELS)
    terrain.terrain_types[:] = 0
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
    origins = terrain.terrain_origins.detach().cpu().numpy()
    model = env.sim._mj_model
    geoms = _geoms(model)
    sx = float(cfg.scene.terrain.terrain_generator.size[0])

    fig, axes = plt.subplots(LEVELS, 1, figsize=(13, 1.25 * LEVELS), sharex=True, constrained_layout=True)
    for lv, ax in enumerate(axes):
        ox, oy, _ = origins[lv, 0]
        for x, y, hx, hy in geoms:
            dx, dy = x - ox, y - oy
            if not (-0.6 <= dx <= sx + 0.6 and abs(dy) <= 0.9):
                continue
            if hx > 1.0:
                color = "#555555"          # 沟底 / 地面（整块）
                ax.add_patch(
                    patches.Rectangle((dx - hx, dy - hy), 2 * hx, 2 * hy,
                                      facecolor=color, edgecolor="none", zorder=1)
                )
                continue
            if hy > 0.5:
                color = COLORS["slab"][0]   # 平台（整宽）
            elif hx <= 0.021:
                color = COLORS["waypoint"][0]
            else:
                color = COLORS["pillar"][0]
            ax.add_patch(
                patches.Rectangle((dx - hx, dy - hy), 2 * hx, 2 * hy,
                                  facecolor=color, edgecolor="k", lw=0.3, zorder=3)
            )
        ax.set_xlim(-0.5, sx + 0.1)
        ax.set_ylim(-0.8, 0.8)
        ax.set_yticks([])
        ax.set_ylabel(f"L{lv}", rotation=0, labelpad=14, fontsize=9)
        ax.grid(True, axis="x", lw=0.3, alpha=0.5)

    axes[-1].set_xlabel("x [m] (spawn = 0)")
    fig.suptitle(title, fontsize=12)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    env.close()
    print(f"wrote {out}")


def render_combo(out: pathlib.Path) -> None:
    """两条赛道并排的地图：一行一个难度，上下并排两列。"""
    task = "Robot-ObstacleCombo-v0"
    cfg = load_env_cfg(task)
    cfg.scene.num_envs = LEVELS * 2
    cfg.scene.terrain.num_envs = LEVELS * 2
    cfg.scene.terrain.terrain_generator.num_rows = LEVELS
    cfg.scene.terrain.terrain_generator.num_cols = 2
    env = ManagerBasedRlEnv(cfg, device="cpu")
    env.reset()
    origins = env.scene.terrain.terrain_origins.detach().cpu().numpy()
    geoms = _geoms(env.sim._mj_model)
    sx = float(cfg.scene.terrain.terrain_generator.size[0])

    fig, axes = plt.subplots(LEVELS, 1, figsize=(13, 1.4 * LEVELS), sharex=True, constrained_layout=True)
    for lv, ax in enumerate(axes):
        for col in range(origins.shape[1]):
            ox, oy, _ = origins[lv, col]
            y_shift = col * 1.9
            for x, y, hx, hy in geoms:
                dx, dy = x - ox, y - oy
                if not (-0.6 <= dx <= sx + 0.6 and abs(dy) <= 0.9):
                    continue
                if hx > 1.0:
                    color = "#555555"
                elif hy > 0.5:
                    color = COLORS["slab"][0]
                elif hx <= 0.021:
                    color = COLORS["waypoint"][0]
                else:
                    color = COLORS["pillar"][0]
                ax.add_patch(
                    patches.Rectangle((dx - hx, dy - hy + y_shift), 2 * hx, 2 * hy,
                                      facecolor=color, edgecolor="k", lw=0.3, zorder=3)
                )
        ax.set_xlim(-0.5, sx + 0.1)
        ax.set_ylim(-1.0, 2.8)
        ax.set_yticks([])
        ax.set_ylabel(f"L{lv}", rotation=0, labelpad=14, fontsize=9)
        ax.grid(True, axis="x", lw=0.3, alpha=0.5)

    axes[-1].set_xlabel("x [m] (spawn = 0); 上=绕桩列, 下=沟壑列")
    fig.suptitle("Combo: gap course + slalom side by side in ONE scene", fontsize=12)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    env.close()
    print(f"wrote {out}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="docs/images")
    args = parser.parse_args()
    out = pathlib.Path(args.out_dir)
    render("Robot-GapCourse-v0", out / "gapcourse_levels.png", "Gap course (slabs 0.4 m + gap 0.05..0.45 m)")
    render("Robot-Slalom-v0", out / "slalom_levels.png", "Slalom (pillars 50x50 mm + alternating waypoints)")
    render_combo(out / "combo_levels.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
