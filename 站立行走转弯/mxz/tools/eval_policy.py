"""评估训练好的策略：把速度指令钉死成固定值，测量跟踪误差与 base 高度。

用法::

    python tools/eval_policy.py --task Robot-WalkTurn-Stand-v0 \
        --checkpoint checkpoints/model_299.pt

    python tools/eval_policy.py --task Robot-WalkTurn-v0 \
        --checkpoint checkpoints/model_1795.pt \
        --scenario straight_fast,straight_slow,turn_left,turn_right,arc_left

    # 自己训练出来的 run 也能用：
    python tools/eval_policy.py --task Robot-WalkTurn-Locked-v0 \
        --checkpoint logs/rsl_rl/robot_walk_turn_locked/<run>/model_1795.pt

做两件事：
1. 指标：对每个场景把指令写死，跑固定时长，统计实际速度 vs 指令速度、base 高度。
2. 视频（--video）：每个场景用 1 个环境渲染一段 mp4，存到 docs/videos/。

注意：mjlab 的指令项每 10 s 会重采样，这里把 term._resample_command 打成空函数，
并在每一步之前把 vel_command_b 重新写一遍。
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg  # noqa: E402
from mjlab.utils.torch import configure_torch_backends  # noqa: E402
from mjlab.utils.wrappers import VideoRecorder  # noqa: E402

import robot  # noqa: E402,F401  注册任务

COMMAND_NAME = "twist"

# (名字, vx, vy, wz)
SCENARIOS: list[tuple[str, float, float, float]] = [
    ("straight_fast", 0.8, 0.0, 0.0),
    ("straight_slow", 0.4, 0.0, 0.0),
    ("backward", -0.4, 0.0, 0.0),
    ("turn_left", 0.0, 0.0, 0.8),
    ("turn_right", 0.0, 0.0, -0.8),
    ("arc_left", 0.6, 0.0, 0.6),
    ("lateral", 0.0, 0.3, 0.0),
    ("stand", 0.0, 0.0, 0.0),
]


def make_env(task: str, num_envs: int, device: str, render_mode: str | None = None):
    cfg = load_env_cfg(task, play=True)
    cfg.scene.num_envs = num_envs
    if cfg.scene.terrain is not None:
        cfg.scene.terrain.num_envs = num_envs
    cfg.viewer.width = 960
    cfg.viewer.height = 640
    if render_mode == "rgb_array":
        viewer_cls = type(cfg.viewer)
        cfg.viewer.origin_type = viewer_cls.OriginType.ASSET_ROOT
        cfg.viewer.entity_name = "robot"
        cfg.viewer.lookat = (0.0, 0.0, 0.18)
        cfg.viewer.distance = 1.7
        cfg.viewer.elevation = -5.0
        cfg.viewer.azimuth = 90.0
        cfg.viewer.max_extra_envs = 0
    env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode=render_mode)
    return cfg, env


def load_policy(task: str, wrapped, device: str, checkpoint: Path):
    agent_cfg = load_rl_cfg(task)
    runner = MjlabOnPolicyRunner(wrapped, asdict(agent_cfg), device=device)
    runner.load(
        str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device
    )
    return runner.get_inference_policy(device=device)


def pin_command(env: ManagerBasedRlEnv, cmd: tuple[float, float, float]) -> None:
    term = env.command_manager.get_term(COMMAND_NAME)
    if not getattr(term, "_command_pinned", False):
        term._resample_command = lambda env_ids: None  # type: ignore[method-assign]
        term._command_pinned = True
    term.is_standing_env[:] = False
    term.is_heading_env[:] = False
    term.is_world_env[:] = False
    vec = torch.tensor(cmd, dtype=torch.float32, device=env.device)
    term.vel_command_b[:] = vec
    term.vel_command_w[:] = vec


def rollout(env, wrapped, policy, cmd, steps: int, settle: int = 25):
    """跑 steps+settle 步，返回 settle 之后的实际速度/高度统计。"""
    robot = env.scene["robot"]
    obs, _ = wrapped.reset()
    pin_command(env, cmd)

    lin_hist, ang_hist, z_hist, up_hist, done_count = [], [], [], [], 0
    for i in range(steps + settle):
        pin_command(env, cmd)
        with torch.no_grad():
            actions = policy(obs)
        obs, _rew, dones, _extras = wrapped.step(actions)
        done_count += int(dones.sum().item())
        if i >= settle:
            lin_hist.append(robot.data.root_link_lin_vel_b[:, :2].clone())
            ang_hist.append(robot.data.root_link_ang_vel_b[:, 2].clone())
            z_hist.append(robot.data.root_link_pos_w[:, 2].clone())
            up_hist.append((-robot.data.projected_gravity_b[:, 2]).clone())

    lin = torch.stack(lin_hist).mean(dim=0)  # (num_envs, 2)
    ang = torch.stack(ang_hist).mean(dim=0)  # (num_envs,)
    z_t = torch.stack(z_hist)  # (steps, num_envs)
    up_t = torch.stack(up_hist)  # (steps, num_envs)
    z = z_t.mean(dim=0)
    return {
        "vx": lin[:, 0].mean().item(),
        "vy": lin[:, 1].mean().item(),
        "wz": ang.mean().item(),
        "z": z.mean().item(),
        "z_std": z_t.std(dim=0).mean().item(),
        "z_min": z_t.min().item(),
        "z_max": z_t.max().item(),
        "up_mean": up_t.mean().item(),
        "up_min": up_t.min().item(),
        "dones": done_count,
        "vx_absdev": (lin[:, 0] - cmd[0]).abs().mean().item(),
        "vy_absdev": (lin[:, 1] - cmd[1]).abs().mean().item(),
        "wz_absdev": (ang - cmd[2]).abs().mean().item(),
    }


def evaluate(args, env, wrapped, policy) -> None:
    print()
    print("=" * 108)
    print(
        f"{'scenario':<14}{'cmd vx/vy/wz':>22}{'achieved vx/vy/wz':>24}"
        f"{'|err| xy':>10}{'|err| yaw':>11}{'base z':>9}{'z std':>8}{'z min/max':>14}"
        f"{'up mean/min':>14}{'done':>6}"
    )
    print("-" * 122)
    for name, vx, vy, wz in SCENARIOS:
        r = rollout(env, wrapped, policy, (vx, vy, wz), args.steps, settle=args.settle)
        cmd_str = f"{vx:+.2f} / {vy:+.2f} / {wz:+.2f}"
        got_str = f"{r['vx']:+.3f} / {r['vy']:+.3f} / {r['wz']:+.3f}"
        xy_err = r["vx_absdev"] + r["vy_absdev"]
        z_range = f"{r['z_min']:.3f}/{r['z_max']:.3f}"
        up_str = f"{r['up_mean']:.3f}/{r['up_min']:.3f}"
        print(
            f"{name:<14}{cmd_str:>22}{got_str:>24}"
            f"{xy_err:>10.3f}{r['wz_absdev']:>11.3f}"
            f"{r['z']:>9.3f}{r['z_std']:>8.3f}{z_range:>14}"
            f"{up_str:>14}{r['dones']:>6}"
        )
    print("=" * 122)
    print(
        "说明：|err| xy = |实际 vx - 指令 vx| + |实际 vy - 指令 vy| 的均值；"
        "|err| yaw = |实际 wz - 指令 wz|；done = 终止次数（摔倒/超时）。"
    )


def record_videos(args, task: str, device: str, checkpoint: Path) -> None:
    cfg, env = make_env(task, 1, device, render_mode="rgb_array")
    video_length = args.steps
    recorder = VideoRecorder(
        env,
        video_folder=ROOT / "docs" / "videos",
        step_trigger=lambda step: step % video_length == 0,
        video_length=video_length,
        disable_logger=True,
    )
    wrapped = RslRlVecEnvWrapper(recorder, clip_actions=load_rl_cfg(task).clip_actions)
    policy = load_policy(task, wrapped, device, checkpoint)

    print()
    print(f"录制视频到 {ROOT / 'docs' / 'videos'}（每段 {video_length} 步）")
    for name, vx, vy, wz in SCENARIOS:
        recorder.name_prefix = f"policy_{name}"
        rollout(recorder, wrapped, policy, (vx, vy, wz), video_length, settle=0)
        print(f"  {name:<14} -> {recorder.name_prefix}-step-{recorder.step_count - video_length}.mp4")
    recorder.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="评估 Robot-WalkTurn 策略")
    parser.add_argument("--checkpoint", required=True, help="model_XXXX.pt 的路径")
    parser.add_argument("--task", default="Robot-WalkTurn-v0")
    parser.add_argument("--num-envs", type=int, default=16, help="并行环境数（指标统计用）")
    parser.add_argument("--steps", type=int, default=250, help="每个场景运行的环境步数（50 Hz，250 步 = 5 s）")
    parser.add_argument("--settle", type=int, default=50, help="先跑多少步再统计（50 步 = 1 s）")
    parser.add_argument("--scenario", default="all", help="逗号分隔的场景名，默认 all")
    parser.add_argument("--device", default=None)
    parser.add_argument("--video", action="store_true", help="额外录制每个场景的视频")
    args = parser.parse_args()

    global SCENARIOS
    if args.scenario.lower() != "all":
        wanted = {s.strip() for s in args.scenario.split(",") if s.strip()}
        SCENARIOS = [s for s in SCENARIOS if s[0] in wanted]
        if not SCENARIOS:
            print(f"[错误] 没有匹配的场景: {args.scenario}")
            return 1

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = (ROOT / checkpoint).resolve()
    if not checkpoint.exists():
        print(f"[错误] 找不到 checkpoint: {checkpoint}")
        return 1

    configure_torch_backends()
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] task: {args.task}  checkpoint: {checkpoint.name}  device: {device}")

    cfg, env = make_env(args.task, args.num_envs, device)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(args.task).clip_actions)
    policy = load_policy(args.task, wrapped, device, checkpoint)
    try:
        evaluate(args, env, wrapped, policy)
    finally:
        env.close()

    if args.video:
        record_videos(args, args.task, device, checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())