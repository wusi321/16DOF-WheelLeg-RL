"""Smoke test: build the go2w task envs, check the obs/action contract, step them.

Runs on CPU too (mujoco-warp compiles its kernels for the CPU device, slowly).
    uv run python tools/smoke_build_env.py
"""

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

TASKS = (
    "Robot-Flat-v0",
    "Robot-Rough-v0",
    "Robot-Crawl-v0",
    "Robot-StepStone-v0",
)

#: 零动作“站立”回归检查：轮子必须和地面接触，否则机器人会穿地。
#: （ccd_iterations=0 时 MuJoCo 的网格碰撞会失效，症状就是每个 episode
#:   只有十几步就因为 base 触地而终止。）
STAND_STEPS = 100
#: 每个任务零动作站稳时的最低机身高度（crawl 是蹲姿任务，天然低）
STAND_MIN_HEIGHT = {
    "Robot-Flat-v0": 0.25,
    "Robot-Rough-v0": 0.25,
    "Robot-Crawl-v0": 0.06,
    "Robot-StepStone-v0": 0.25,
}

failed = []

for task_id in TASKS:
    cfg = load_env_cfg(task_id)
    cfg.scene.num_envs = 2
    cfg.scene.terrain.num_envs = 2
    if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.num_rows = 2
        cfg.scene.terrain.terrain_generator.num_cols = 2
    print(f"\n===== {task_id} =====")
    print("episode_length_s:", cfg.episode_length_s, "decimation:", cfg.decimation)
    env = ManagerBasedRlEnv(cfg, device="cpu")
    print("action dim:", env.action_manager.total_action_dim)
    for name, term in env.action_manager._terms.items():
        print(f"  action term '{name}' ({term.action_dim}): {term.target_names}")
    for group, dim in env.observation_manager.group_obs_dim.items():
        print(f"  obs group '{group}': {dim}")
    print("  reward terms:", len(env.reward_manager.active_terms), "| termination terms:",
          len(env.termination_manager.active_terms))

    obs, _ = env.reset()
    print("reset ok; actor:", tuple(obs["actor"].shape), "critic:", tuple(obs["critic"].shape))

    robot = env.scene["robot"]
    origins = env.scene.env_origins.clone()
    actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim)
    rew_sum = 0.0
    for _ in range(10):
        obs, rew, term, trunc, _ = env.step(actions)
        rew_sum += float(rew.mean())
    print(f"10 steps ok; mean reward/step {rew_sum / 10:+.4f}; finite: "
          f"{bool(torch.isfinite(obs['actor']).all())}; "
          f"terminated: {int(term.sum())}, truncated: {int(trunc.sum())}")

    # 站立回归：继续零动作，机器人必须站得住（轮子接触地面）
    term_total = 0
    for _ in range(STAND_STEPS):
        obs, rew, term, trunc, _ = env.step(actions)
        term_total += int(term.sum())
    heights = (robot.data.root_link_pos_w[:, 2] - origins[:, 2]).tolist()
    min_h = min(heights)
    limit = STAND_MIN_HEIGHT.get(task_id, 0.2)
    ok = term_total == 0 and min_h > limit
    print(f"  stand {STAND_STEPS} steps: {'OK' if ok else 'FAILED'}; "
          f"terminations={term_total}; base_z={[round(h, 3) for h in heights]}")
    if not ok:
        failed.append(task_id)
    env.close()

if failed:
    print(f"\n站立回归失败: {failed}")
    raise SystemExit(1)
print("\nall tasks ok")
