"""PPO 训练超参（RSL-RL）。数值沿用旧 WalkTurn 工程调好的那套。"""

from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


def walk_turn_ppo_runner_cfg(experiment_name: str = "robot_walk_turn") -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        logger="tensorboard",
        experiment_name=experiment_name,
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=False,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "log",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=False,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.003,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=8e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        save_interval=100,
        num_steps_per_env=24,
        max_iterations=10_000,
    )


def walk_turn_play_runner_cfg(experiment_name: str = "robot_walk_turn_play") -> RslRlOnPolicyRunnerCfg:
    return walk_turn_ppo_runner_cfg(experiment_name)