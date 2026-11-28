"""Evaluate the hybrid controller:

* straight mode (|wz| < 0.1): stand policy holds the 45 deg legs, wheels are
  driven at vx / R with a 1.14 slip feed-forward;
* turn mode (|wz| >= 0.1): the locked-turn policy controls the legs and the
  wheel action is forced to zero.

The environment is the generic ``Robot-WalkTurn-v0`` play env (wheel scale 10),
so the raw wheel action written here is the final target divided by 10.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_policy import (  # noqa: E402
    VideoRecorder,
    load_policy,
    make_env,
    pin_command,
)

from robot import naming  # noqa: E402

SLIP = 1.14
WHEEL_SCALE = 10.0


def _joint_ids(robot, device):
    names = list(robot.joint_names)
    wheels = torch.tensor(
        [i for i, n in enumerate(names) if "wheel" in n.lower()],
        dtype=torch.long,
        device=device,
    )
    legs = torch.tensor(
        [i for i, n in enumerate(names) if "wheel" not in n.lower()],
        dtype=torch.long,
        device=device,
    )
    return legs, wheels


def _select(obs, cmd, locked_policy):
    """Return the 16-d action for one command.

    Straight mode deliberately uses zero leg action: the 45 deg stance is
    held by the joint PD, so the legs do not step.  This is the behavior that
    produced the recorded hybrid results.
    """
    actions = torch.zeros(obs.shape[0], 16, device=obs.device)
    if abs(cmd[2]) < 0.1:
        target = (cmd[0] / naming.WHEEL_RADIUS) * SLIP
        actions[:, 12:16] = target / WHEEL_SCALE
    else:
        with torch.no_grad():
            actions[:] = locked_policy(obs)
        actions[:, 12:16] = 0.0
    return actions


def rollout(env, wrapped, locked_policy, cmd, steps, settle=50):
    robot = env.scene["robot"]
    legs, wheels = _joint_ids(robot, env.device)
    obs, _ = wrapped.reset()
    pin_command(env, cmd)
    vx_hist, wz_hist, z_hist, leg_hist, wheel_hist, up_hist = [], [], [], [], [], []
    dones = 0
    for i in range(steps):
        pin_command(env, cmd)
        actions = _select(obs, cmd, locked_policy)
        obs, _rew, done, _extra = wrapped.step(actions)
        dones += int(done.sum().item())
        if i >= settle:
            vx_hist.append(robot.data.root_link_lin_vel_b[:, 0].clone())
            wz_hist.append(robot.data.root_link_ang_vel_b[:, 2].clone())
            z_hist.append(robot.data.root_link_pos_w[:, 2].clone())
            leg_hist.append(robot.data.joint_vel[:, legs].abs().mean(dim=1).clone())
            wheel_hist.append(robot.data.joint_vel[:, wheels].abs().mean(dim=1).clone())
            up_hist.append((-robot.data.projected_gravity_b[:, 2]).clone())

    def avg(hist):
        return torch.stack(hist).mean().item()

    return {
        "vx": avg(vx_hist),
        "wz": avg(wz_hist),
        "z": avg(z_hist),
        "leg": avg(leg_hist),
        "wheel": avg(wheel_hist),
        "up": avg(up_hist),
        "done": dones,
    }


def transition(env, wrapped, locked_policy, steps_each=100):
    """straight 0.4 -> turn +0.8 -> straight 0.4, hard switch."""
    robot = env.scene["robot"]
    obs, _ = wrapped.reset()
    phases = [
        ("straight", (0.4, 0.0, 0.0)),
        ("turn", (0.0, 0.0, 0.8)),
        ("straight", (0.4, 0.0, 0.0)),
    ]
    dones = 0
    zs = []
    for _name, cmd in phases:
        pin_command(env, cmd)
        for _ in range(steps_each):
            actions = _select(obs, cmd, locked_policy)
            obs, _rew, done, _extra = wrapped.step(actions)
            dones += int(done.sum().item())
            zs.append(robot.data.root_link_pos_w[:, 2].clone())
    z = torch.stack(zs)
    return {
        "done": dones,
        "z_mean": z.mean().item(),
        "z_min": z.min().item(),
        "z_max": z.max().item(),
    }


def record_videos(args, locked_policy, device):
    """Record one short clip per mode with a separate 1-env recorder."""
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_rl_cfg

    _, env = make_env("Robot-WalkTurn-v0", 1, device, render_mode="rgb_array")
    recorder = VideoRecorder(
        env,
        video_folder=ROOT / "docs" / "videos",
        step_trigger=lambda step: step % args.steps == 0,
        video_length=args.steps,
        disable_logger=True,
    )
    wrapped = RslRlVecEnvWrapper(
        recorder, clip_actions=load_rl_cfg("Robot-WalkTurn-v0").clip_actions
    )
    for name, cmd in (
        ("stand", (0.0, 0.0, 0.0)),
        ("straight_0.4", (0.4, 0.0, 0.0)),
        ("straight_0.8", (0.8, 0.0, 0.0)),
        ("turn_left", (0.0, 0.0, 0.8)),
        ("turn_right", (0.0, 0.0, -0.8)),
    ):
        recorder.name_prefix = f"hybrid_{name}"
        rollout(recorder, wrapped, locked_policy, cmd, args.steps, settle=0)
        print(
            f"  video -> docs/videos/{recorder.name_prefix}-step-"
            f"{recorder.step_count - args.steps}.mp4"
        )
    recorder.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--locked-checkpoint",
        default="checkpoints/model_1795.pt",
        help="bundled locked-wheel turn policy (override after training your own run)",
    )
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_rl_cfg
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends()
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    locked_ckpt = Path(args.locked_checkpoint)
    if not locked_ckpt.is_absolute():
        locked_ckpt = ROOT / locked_ckpt
    if not locked_ckpt.exists():
        print(f"[error] checkpoint not found: {locked_ckpt}")
        print("        the repo ships checkpoints/model_1795.pt")
        return 1

    # Generic play env; wheel action scale is 10 here, so we write target/10.
    _, env = make_env("Robot-WalkTurn-v0", args.num_envs, device)
    wrapped = RslRlVecEnvWrapper(
        env, clip_actions=load_rl_cfg("Robot-WalkTurn-v0").clip_actions
    )
    locked_policy = load_policy(
        "Robot-WalkTurn-Locked-v0", wrapped, device, locked_ckpt
    )

    print(f"hybrid: slip={SLIP}, locked={locked_ckpt}")
    print("scenario        cmd vx/wz       achieved vx/wz   base z  leg|v| wheel|v| up    done")
    scenarios = [
        ("stand", (0.0, 0.0, 0.0)),
        ("straight_0.2", (0.2, 0.0, 0.0)),
        ("straight_0.4", (0.4, 0.0, 0.0)),
        ("straight_0.8", (0.8, 0.0, 0.0)),
        ("turn_left", (0.0, 0.0, 0.8)),
        ("turn_right", (0.0, 0.0, -0.8)),
    ]
    for name, cmd in scenarios:
        r = rollout(env, wrapped, locked_policy, cmd, args.steps)
        print(
            f"{name:14s} {cmd[0]:+.2f}/{cmd[2]:+.2f}  "
            f"{r['vx']:+.3f}/{r['wz']:+.3f}  {r['z']:.3f}  "
            f"{r['leg']:.3f}  {r['wheel']:.2f}  {r['up']:.3f}  {r['done']}"
        )

    t = transition(env, wrapped, locked_policy)
    print(
        f"transition straight->turn->straight: done={t['done']} "
        f"z={t['z_mean']:.3f} [{t['z_min']:.3f},{t['z_max']:.3f}]"
    )
    if args.video:
        print("recording hybrid videos ...")
        record_videos(args, locked_policy, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())