"""用固定侧视角把策略跑一段并录成 mp4（带地面，方便看清姿态）。

与 eval_policy.py 的区别：eval_policy 的相机绑定在机器人上且不带地面参照；
这里用 MuJoCo 自由相机，每帧把 lookat 跟到机器人 xy，距离/方位固定。

用法::

    python tools/render_policy.py --checkpoint logs/.../model_998.pt \
        --scenario stand,straight_fast,turn_left,turn_right --steps 250
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio.v2 as iio
import mujoco
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from eval_policy import SCENARIOS, load_policy, make_env, pin_command  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402

from robot.config.rl_cfg import walk_turn_ppo_runner_cfg  # noqa: E402,F401
from mjlab.tasks.registry import load_rl_cfg  # noqa: E402
from mjlab.utils.torch import configure_torch_backends  # noqa: E402


def render_scenario(env, wrapped, policy, renderer, name, cmd, steps, out_dir):
    robot = env.scene["robot"]
    obs, _ = wrapped.reset()
    pin_command(env, cmd)
    frames = []
    for _ in range(steps):
        pin_command(env, cmd)
        with torch.no_grad():
            actions = policy(obs)
        obs, _rew, _dones, _extras = wrapped.step(actions)
        xy = env.sim.mj_data.qpos[0:2]
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = (float(xy[0]), float(xy[1]), 0.24)
        cam.distance = 2.1
        cam.azimuth = 135.0
        cam.elevation = -12.0
        renderer.update_scene(env.sim.mj_data, camera=cam)
        frames.append(renderer.render())
    path = out_dir / f"fixed_{name}.mp4"
    iio.mimsave(path, frames, fps=50, quality=8)
    print(f"  {name:<14} -> {path.name}  ({len(frames)} frames)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", default="Robot-WalkTurn-v0")
    parser.add_argument("--scenario", default="stand,straight_fast,turn_left,turn_right")
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    configure_torch_backends()
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt = Path(args.checkpoint)
    if not ckpt.is_absolute():
        ckpt = (ROOT / ckpt).resolve()
    if not ckpt.exists():
        print(f"[错误] 找不到 checkpoint: {ckpt}")
        return 1

    wanted = {s.strip() for s in args.scenario.split(",") if s.strip()}
    scenarios = [s for s in SCENARIOS if s[0] in wanted]
    if not scenarios:
        print("[错误] 没有匹配的场景")
        return 1

    cfg, env = make_env(args.task, 1, device)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(args.task).clip_actions)
    policy = load_policy(args.task, wrapped, device, ckpt)
    renderer = mujoco.Renderer(env.sim.mj_model, height=480, width=640)
    out_dir = ROOT / "docs" / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] task={args.task} checkpoint={ckpt.name} device={device}")
    try:
        for name, vx, vy, wz in scenarios:
            render_scenario(env, wrapped, policy, renderer, name, (vx, vy, wz), args.steps, out_dir)
    finally:
        renderer.close()
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())