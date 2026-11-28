"""One-iteration PPO smoke test (CPU friendly).

Validates the full training path without a launch wrapper: env build -> rollout
-> GAE -> PPO update -> tensorboard logging.  Useful to check a fresh checkout
before launching a long GPU run.

    uv run python tools/smoke_train_ppo.py
"""

from dataclasses import asdict

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

TASK = "Robot-Flat-v0"

cfg = load_env_cfg(TASK)
cfg.scene.num_envs = 2
cfg.scene.terrain.num_envs = 2
cfg.episode_length_s = 1.0

agent = load_rl_cfg(TASK)
agent.max_iterations = 1
agent.num_steps_per_env = 4
agent.save_interval = 1000
agent.seed = 0
agent.logger = "tensorboard"

env = ManagerBasedRlEnv(cfg, device="cpu")
env = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)

runner = MjlabOnPolicyRunner(env, asdict(agent), "logs/smoke_ppo", "cpu")
runner.learn(num_learning_iterations=1, init_at_random_ep_len=True)
print("PPO iteration OK")
env.close()
