from mjlab.tasks.registry import register_mjlab_task
from .config.env_cfgs import (
    flat_env_cfg,
    rough_env_cfg,
    crawl_env_cfg,
    stepstone_env_cfg,
    gapcourse_env_cfg,
    slalom_env_cfg,
)
from .config.rl_cfg import (
    flat_ppo_runner_cfg,
    rough_ppo_runner_cfg,
    crawl_ppo_runner_cfg,
    stepstone_ppo_runner_cfg,
    gapcourse_ppo_runner_cfg,
    slalom_ppo_runner_cfg,
)

register_mjlab_task(
    task_id="Robot-Flat-v0",
    env_cfg=flat_env_cfg(),
    play_env_cfg=flat_env_cfg(play=True),
    rl_cfg=flat_ppo_runner_cfg(),
)

register_mjlab_task(
    task_id="Robot-Rough-v0",
    env_cfg=rough_env_cfg(),
    play_env_cfg=rough_env_cfg(play=True),
    rl_cfg=rough_ppo_runner_cfg(),
)

register_mjlab_task(
    task_id="Robot-Crawl-v0",
    env_cfg=crawl_env_cfg(),
    play_env_cfg=crawl_env_cfg(play=True),
    rl_cfg=crawl_ppo_runner_cfg(),
)

# 梅花桩（Robocon2027 stepping_stones 段），课程/奖励见 16DOF_RL.md
register_mjlab_task(
    task_id="Robot-StepStone-v0",
    env_cfg=stepstone_env_cfg(),
    play_env_cfg=stepstone_env_cfg(play=True),
    rl_cfg=stepstone_ppo_runner_cfg(),
)

# 沟壑课程：40 cm 平台 + 5..45 cm 沟，奖励跃沟
register_mjlab_task(
    task_id="Robot-GapCourse-v0",
    env_cfg=gapcourse_env_cfg(),
    play_env_cfg=gapcourse_env_cfg(play=True),
    rl_cfg=gapcourse_ppo_runner_cfg(),
)

# 绕桩课程：一排方柱 + 左右交替必经点，必须按顺序经过
register_mjlab_task(
    task_id="Robot-Slalom-v0",
    env_cfg=slalom_env_cfg(),
    play_env_cfg=slalom_env_cfg(play=True),
    rl_cfg=slalom_ppo_runner_cfg(),
)
