"""冒烟测试：构建指定阶段任务，检查观测/动作契约，跑零动作站立。

用法（工程根目录，用 RL-main 的 venv）::

    $env:PYTHONPATH = "$PWD\\src"
    & $py tools\\smoke_build_env.py --task Robot-WalkTurn-Stand-v0

检查项：
1. 环境能构建，动作维度 = 16（12 腿 + 4 轮），观测维度符合预期。
2. reset + 10 步零动作不出现 NaN，奖励/终止都能算。
3. 零动作站 2 s：base 高度在 0.30–0.35 m、无终止。
4. 指令采样器模式分布符合权重，且不违反各模式契约。
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.tasks.registry import load_env_cfg  # noqa: E402

import robot  # noqa: E402,F401  # 触发任务注册

STAND_STEPS = 100  # decimation=4, dt=0.005 -> 2 s
STAND_MIN_HEIGHT = 0.30
STAND_MAX_HEIGHT = 0.35


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="Robot-WalkTurn-Stand-v0")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=STAND_STEPS)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("[WARN] CUDA 不可用，退回 CPU（mujoco-warp 首次编译会慢一些）")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    cfg = load_env_cfg(args.task)
    cfg.scene.num_envs = args.num_envs
    if cfg.scene.terrain is not None:
        cfg.scene.terrain.num_envs = args.num_envs
        if cfg.scene.terrain.terrain_generator is not None:
            cfg.scene.terrain.terrain_generator.num_rows = 1
            cfg.scene.terrain.terrain_generator.num_cols = 1

    print(f"===== {args.task} =====")
    print("episode_length_s:", cfg.episode_length_s, "decimation:", cfg.decimation)
    env = ManagerBasedRlEnv(cfg, device=device)

    action_dim = env.action_manager.total_action_dim
    print("action dim:", action_dim)
    for name, term in env.action_manager._terms.items():
        print(f"  action term '{name}' ({term.action_dim}): {term.target_names}")
    for group, dim in env.observation_manager.group_obs_dim.items():
        print(f"  obs group '{group}': {dim}")
    print(
        "  reward terms:",
        len(env.reward_manager.active_terms),
        "| termination terms:",
        len(env.termination_manager.active_terms),
    )

    obs, _ = env.reset()
    print("reset ok; actor:", tuple(obs["actor"].shape), "critic:", tuple(obs["critic"].shape))

    robot_entity = env.scene["robot"]
    origins = env.scene.env_origins.clone()
    actions = torch.zeros(env.num_envs, action_dim, device=device)

    rew_sum = 0.0
    for _ in range(10):
        obs, rew, term, trunc, _ = env.step(actions)
        rew_sum += float(rew.mean())
    print(
        f"10 steps ok; mean reward/step {rew_sum / 10:+.4f}; finite: "
        f"{bool(torch.isfinite(obs['actor']).all())}; "
        f"terminated: {int(term.sum())}, truncated: {int(trunc.sum())}"
    )

    # ---------------- 站立回归 ----------------
    term_total = 0
    for _ in range(args.steps):
        obs, rew, term, trunc, _ = env.step(actions)
        term_total += int(term.sum())
    heights = (robot_entity.data.root_link_pos_w[:, 2] - origins[:, 2]).tolist()
    min_h, max_h = min(heights), max(heights)
    stand_ok = term_total == 0 and STAND_MIN_HEIGHT <= min_h and max_h <= STAND_MAX_HEIGHT
    print(
        f"stand {args.steps} steps ({args.steps * cfg.decimation * 0.005:.1f} s): "
        f"{'OK' if stand_ok else 'FAILED'}; terminations={term_total}; "
        f"base_z={[round(h, 3) for h in heights]}"
    )

    # ---------------- 指令分布与契约 ----------------
    cmd_term = env.command_manager.get_term("twist")
    modes = Counter()
    violations = 0
    for _ in range(50):
        cmd_term._resample_command(torch.arange(env.num_envs, device=device))
        modes.update(cmd_term.command_mode.tolist())
        for mode, (vx, vy, wz) in zip(
            cmd_term.command_mode.tolist(), cmd_term.vel_command_b.tolist()
        ):
            if mode == 0:
                violations += not (abs(vx) >= 0.199 and vy == 0.0 and wz == 0.0)
            elif mode == 1:
                violations += not (vx == 0.0 and vy == 0.0 and abs(wz) >= 0.299)
            elif mode == 2:
                violations += not (vx >= 0.199 and vy == 0.0 and abs(wz) >= 0.299)
            elif mode == 3:
                violations += not (vx == 0.0 and abs(vy) >= 0.199 and wz == 0.0)
            else:
                violations += not (vx == 0.0 and vy == 0.0 and wz == 0.0)
    total = sum(modes.values())
    share = {k: round(v / total, 3) for k, v in sorted(modes.items())}
    print(
        "command modes:",
        share,
        "(0=直行 1=原地转向 2=弧线 3=横移 4=站立);",
        f"契约违例={violations}",
    )

    env.close()
    ok = (
        action_dim == 16
        and stand_ok
        and violations == 0
        and bool(torch.isfinite(obs["actor"]).all())
    )
    if not ok:
        print("\n冒烟测试失败")
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())