"""Live hybrid controller in the native MuJoCo (mjlab scene) viewer.

No video, no replay: one robot is simulated in the MuJoCo viewer and the
straight/turn mode is selected exactly like ``eval_hybrid.py``.

Straight: legs hold the 45 deg stance with zero joint action (PD), wheels
          are driven by (vx / R * 1.14) / 10.
Turn:     the locked-wheel gait policy (checkpoints/model_1795.pt) drives the
          legs and the wheel action is forced to zero.

Auto demo (default):
    stand 2 s -> straight 0.4 (4 s) -> turn left +0.8 (3 s)
    -> straight 0.8 (4 s) -> turn right -0.8 (3 s) -> repeat

Keyboard (native viewer window must have focus):
    W / S : vx +- 0.1 m/s, straight mode
    A / D : wz +- 0.2 rad/s, turn mode
    X     : stop (stand)
    R     : resume auto demo
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

from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_rl_cfg  # noqa: E402
from mjlab.viewer import NativeMujocoViewer, ViewerConfig  # noqa: E402
from robot import naming  # noqa: E402

SLIP = 1.14
WHEEL_SCALE = 10.0

KEY_W, KEY_S, KEY_A, KEY_D, KEY_X, KEY_R = 87, 83, 65, 68, 88, 82

# (steps, command) at 50 Hz. 100 steps = 2 s.
SCHEDULE = [
    (100, (0.0, 0.0, 0.0)),
    (200, (0.4, 0.0, 0.0)),
    (150, (0.0, 0.0, 0.8)),
    (200, (0.8, 0.0, 0.0)),
    (150, (0.0, 0.0, -0.8)),
]


class HybridPolicy:
    def __init__(self, env, locked_policy, manual: bool):
        self.env = env
        self.locked_policy = locked_policy
        self.manual = manual
        self.auto = not manual
        self.cmd = (0.0, 0.0, 0.0)
        self.phase = 0
        self.phase_step = 0
        self.total_steps = 0

    def _set_cmd(self, cmd) -> None:
        self.cmd = (float(cmd[0]), float(cmd[1]), float(cmd[2]))
        pin_command(self.env, self.cmd)

    def _advance_auto(self) -> None:
        steps, cmd = SCHEDULE[self.phase]
        if self.phase_step >= steps:
            self.phase = (self.phase + 1) % len(SCHEDULE)
            self.phase_step = 0
            steps, cmd = SCHEDULE[self.phase]
            print(f"[auto] vx={cmd[0]:+.2f} wz={cmd[2]:+.2f}", flush=True)
        self.phase_step += 1
        self._set_cmd(cmd)

    def reset(self) -> None:
        self.phase = 0
        self.phase_step = 0
        self.total_steps = 0
        if self.auto:
            self.cmd = SCHEDULE[0][1]
        pin_command(self.env, self.cmd)

    def on_key(self, key: int) -> None:
        vx, vy, wz = self.cmd
        if key == KEY_W:
            self.auto = False
            vx, wz = min(0.8, vx + 0.1), 0.0
        elif key == KEY_S:
            self.auto = False
            vx, wz = max(0.0, vx - 0.1), 0.0
        elif key == KEY_A:
            self.auto = False
            vx, wz = 0.0, min(0.8, wz + 0.2)
        elif key == KEY_D:
            self.auto = False
            vx, wz = 0.0, max(-0.8, wz - 0.2)
        elif key == KEY_X:
            self.auto = False
            vx, wz = 0.0, 0.0
        elif key == KEY_R:
            self.auto = True
            self.phase = 0
            self.phase_step = 0
        else:
            return
        self._set_cmd((vx, vy, wz))
        print(
            f"[key] vx={vx:+.2f} wz={wz:+.2f} "
            f"mode={'straight' if abs(wz) < 0.1 else 'turn'}  "
            f"(auto={'on' if self.auto else 'off'})",
            flush=True,
        )

    def __call__(self, obs):
        if self.auto:
            self._advance_auto()
        else:
            pin_command(self.env, self.cmd)

        if abs(self.cmd[2]) < 0.1:
            actions = torch.zeros(
                (obs.shape[0], 16), device=obs.device, dtype=torch.float32
            )
            actions[:, 12:16] = (self.cmd[0] / naming.WHEEL_RADIUS * SLIP) / WHEEL_SCALE
        else:
            with torch.no_grad():
                actions = self.locked_policy(obs).clone()
            actions[:, 12:16] = 0.0

        self.total_steps += 1
        return actions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--locked-checkpoint",
        default="checkpoints/model_1795.pt",
        help="bundled locked-wheel turn policy (override after training your own run)",
    )
    parser.add_argument("--manual", action="store_true", help="start in keyboard mode")
    parser.add_argument("--steps", type=int, default=0, help="0 = run until window closes")
    parser.add_argument("--headless", action="store_true", help="no viewer; smoke test")
    parser.add_argument(
        "--distance",
        type=float,
        default=3.0,
        help="camera distance in m; increase if the robot is clipped",
    )
    parser.add_argument("--elevation", type=float, default=-18.0)
    parser.add_argument("--azimuth", type=float, default=135.0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    locked_ckpt = Path(args.locked_checkpoint)
    if not locked_ckpt.is_absolute():
        locked_ckpt = ROOT / locked_ckpt
    if not locked_ckpt.exists():
        print(f"[error] 找不到锁轮策略 checkpoint: {locked_ckpt}")
        print("        仓库自带 checkpoints/model_1795.pt；")
        print("        如果你自己训练过，用 --locked-checkpoint <path> 指定。")
        return 1

    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends()
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    cfg, env = make_env("Robot-WalkTurn-v0", 1, device)
    cfg.viewer.origin_type = ViewerConfig.OriginType.ASSET_ROOT
    cfg.viewer.entity_name = "robot"
    cfg.viewer.lookat = (0.0, 0.0, 0.24)
    cfg.viewer.distance = args.distance
    cfg.viewer.elevation = args.elevation
    cfg.viewer.azimuth = args.azimuth
    cfg.viewer.max_extra_envs = 0

    wrapped = RslRlVecEnvWrapper(
        env, clip_actions=load_rl_cfg("Robot-WalkTurn-v0").clip_actions
    )
    locked_policy = load_policy("Robot-WalkTurn-Locked-v0", wrapped, device, locked_ckpt)
    policy = HybridPolicy(env, locked_policy, manual=args.manual)

    print("[view_hybrid] straight: zero leg action (45 deg PD), wheel drive")
    print("[view_hybrid] turn: locked-wheel gait policy, wheel action 0")

    try:
        if args.headless:
            robot = env.scene["robot"]
            obs, _ = wrapped.reset()
            for i in range(args.steps or 200):
                actions = policy(obs)
                obs, _rew, _done, _extra = wrapped.step(actions)
                if i % 50 == 0:
                    print(
                        f"[headless] {i:4d} cmd=({policy.cmd[0]:+.2f},"
                        f"{policy.cmd[2]:+.2f}) "
                        f"z={robot.data.root_link_pos_w[0, 2].item():.3f} "
                        f"vx={robot.data.root_link_lin_vel_b[0, 0].item():+.3f} "
                        f"wz={robot.data.root_link_ang_vel_b[0, 2].item():+.3f}",
                        flush=True,
                    )
            print("[headless] done")
        else:
            print("[view_hybrid] auto: stand -> straight 0.4 -> turn +0.8 -> straight 0.8 -> turn -0.8")
            print("[view_hybrid] keys: W/S vx, A/D wz, X stop, R auto")
            print(
                "[view_hybrid] if the robot is clipped: press Backspace to reset the "
                "camera, or scroll out / use --distance 3.5"
            )
            viewer = NativeMujocoViewer(wrapped, policy, key_callback=policy.on_key)
            viewer.run(num_steps=(args.steps or None))
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())