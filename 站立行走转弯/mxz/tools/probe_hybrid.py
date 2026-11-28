"""Probe: wheel-only straight (stand legs + wheel drive) and wheel-locked gait turn.

This does not change any training code. It answers two questions before we
redesign the task:
  1. Can the 45-degree stand policy hold its legs while the wheels roll?
  2. Can the current turn policy still turn when the wheel actions are forced to 0?

Both policies are bundled under checkpoints/ (model_299.pt, model_1396.pt);
override with --stand-checkpoint / --turn-checkpoint.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_policy import load_policy, make_env, pin_command  # noqa: E402

from robot import naming  # noqa: E402


def rollout(env, wrapped, policy, cmd, steps: int, wheel_action: float | None) -> dict:
    robot = env.scene["robot"]
    obs, _ = wrapped.reset()
    pin_command(env, cmd)

    # resolve by name so this also works if joint order changes.
    joint_names = list(robot.joint_names)
    wheel_ids = torch.tensor(
        [i for i, n in enumerate(joint_names) if "wheel" in n.lower()],
        dtype=torch.long,
        device=env.device,
    )
    leg_ids = torch.tensor(
        [i for i, n in enumerate(joint_names) if "wheel" not in n.lower()],
        dtype=torch.long,
        device=env.device,
    )

    vx_hist, vy_hist, wz_hist, z_hist = [], [], [], []
    leg_speed_hist, wheel_speed_hist, up_hist = [], [], []
    done_count = 0
    for _ in range(steps):
        pin_command(env, cmd)
        with torch.no_grad():
            actions = policy(obs)
        if wheel_action is not None:
            actions = actions.clone()
            actions[:, -4:] = wheel_action
        obs, _rew, dones, _extras = wrapped.step(actions)
        done_count += int(dones.sum().item())
        vx_hist.append(robot.data.root_link_lin_vel_b[:, 0].clone())
        vy_hist.append(robot.data.root_link_lin_vel_b[:, 1].clone())
        wz_hist.append(robot.data.root_link_ang_vel_b[:, 2].clone())
        z_hist.append(robot.data.root_link_pos_w[:, 2].clone())
        leg_speed_hist.append(robot.data.joint_vel[:, leg_ids].abs().mean(dim=1).clone())
        wheel_speed_hist.append(robot.data.joint_vel[:, wheel_ids].abs().mean(dim=1).clone())
        up_hist.append((-robot.data.projected_gravity_b[:, 2]).clone())

    def mean(hist):
        return torch.stack(hist).mean().item()

    return {
        "vx": mean(vx_hist),
        "vy": mean(vy_hist),
        "wz": mean(wz_hist),
        "z": mean(z_hist),
        "leg_speed": mean(leg_speed_hist),
        "wheel_speed": mean(wheel_speed_hist),
        "up": mean(up_hist),
        "dones": done_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--slip", type=float, default=1.0, help="feed-forward wheel slip compensation")
    parser.add_argument("--stand-checkpoint", default="checkpoints/model_299.pt")
    parser.add_argument("--turn-checkpoint", default="checkpoints/model_1396.pt")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    stand_ckpt = Path(args.stand_checkpoint)
    if not stand_ckpt.is_absolute():
        stand_ckpt = ROOT / stand_ckpt
    turn_ckpt = Path(args.turn_checkpoint)
    if not turn_ckpt.is_absolute():
        turn_ckpt = ROOT / turn_ckpt
    for label, ckpt in (("stand", stand_ckpt), ("turn", turn_ckpt)):
        if not ckpt.exists():
            print(f"[error] {label} checkpoint not found: {ckpt}")
            return 1

    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends()
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    print("[A] stand policy, legs hold 45deg, wheels driven at vx/R")
    _, env = make_env("Robot-WalkTurn-Stand-v0", args.num_envs, device)
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_rl_cfg

    wrapped = RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg("Robot-WalkTurn-Stand-v0").clip_actions)
    policy = load_policy("Robot-WalkTurn-Stand-v0", wrapped, device, stand_ckpt)
    for vx in (0.2, 0.4, 0.8):
        wheel_action = (vx / naming.WHEEL_RADIUS) / 10.0 * args.slip
        r = rollout(env, wrapped, policy, (0.0, 0.0, 0.0), args.steps, wheel_action)
        print(
            f"  vx_cmd={vx:.2f} wheel_raw={wheel_action:+.3f} "
            f"-> vx={r['vx']:+.3f} vy={r['vy']:+.3f} wz={r['wz']:+.3f} "
            f"z={r['z']:.3f} leg|v|={r['leg_speed']:.3f} "
            f"wheel|v|={r['wheel_speed']:.2f} up={r['up']:.3f} done={r['dones']}"
        )

    print()
    print("[B] turn policy, command wz, wheel actions forced to 0 (brake)")
    _, env = make_env("Robot-WalkTurn-v0", args.num_envs, device)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg("Robot-WalkTurn-v0").clip_actions)
    policy = load_policy("Robot-WalkTurn-v0", wrapped, device, turn_ckpt)
    for wz in (-0.8, 0.8):
        r = rollout(env, wrapped, policy, (0.0, 0.0, wz), args.steps, 0.0)
        print(
            f"  wz_cmd={wz:+.2f} -> vx={r['vx']:+.3f} vy={r['vy']:+.3f} "
            f"wz={r['wz']:+.3f} z={r['z']:.3f} leg|v|={r['leg_speed']:.3f} "
            f"wheel|v|={r['wheel_speed']:.2f} up={r['up']:.3f} done={r['dones']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())