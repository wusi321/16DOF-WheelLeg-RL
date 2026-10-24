"""Smoke test: build the go2w task envs, check the obs/action contract, step them.

Runs on CPU too (mujoco-warp compiles its kernels for the CPU device, slowly).
    uv run python tools/smoke_build_env.py
"""

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

for task_id in ("Robot-Flat-v0", "Robot-Rough-v0", "Robot-Crawl-v0"):
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

    actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim)
    rew_sum = 0.0
    for _ in range(10):
        obs, rew, term, trunc, _ = env.step(actions)
        rew_sum += float(rew.mean())
    print(f"10 steps ok; mean reward/step {rew_sum / 10:+.4f}; finite: "
          f"{bool(torch.isfinite(obs['actor']).all())}; "
          f"terminated: {int(term.sum())}, truncated: {int(trunc.sum())}")
    env.close()
