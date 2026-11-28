"""打印 play 环境下 4 个轮子 geom 的实际摩擦参数（验证域随机化取值）。"""

from __future__ import annotations

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import mujoco

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg


def main() -> int:
    cfg = load_env_cfg("Robot-StepStone-v0", play=True)
    cfg.scene.num_envs = 1
    cfg.scene.terrain.num_envs = 1
    env = ManagerBasedRlEnv(cfg, device="cpu")
    env.reset()
    mj = env.sim._mj_model
    names = {
        "FL_wheel_geom",
        "FR_wheel_geom",
        "RL_wheel_geom",
        "RR_wheel_geom",
    }
    friction = env.sim.model.geom_friction
    if hasattr(friction, "detach"):
        friction = friction.detach().cpu()
    for g in range(mj.ngeom):
        name = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_GEOM, g)
        if name and name.split("/")[-1] in names:
            print(f"{name}: friction={[round(float(v), 4) for v in friction[0, g]]}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
