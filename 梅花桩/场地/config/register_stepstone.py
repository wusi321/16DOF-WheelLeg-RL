"""把梅花桩注册进任务清单（摘录自 task_16dof.py 的 register() 函数）。

不注册 → 训练命令会说"不认识 Robot-16DOF-StepStone-v0 这个任务"。
注册 → 菜单上有了这一项，train_16dof.py / play_16dof.py 才能按名字调用。
"""


def register() -> None:
    stepstone_agent = stepstone_ppo_runner_cfg()
    stepstone_agent.experiment_name = "robot_stepstone_16dof_zero4590_v1_calf121_stancefix"
    stepstone_agent.save_interval = 50
    register_mjlab_task(
        task_id="Robot-16DOF-StepStone-v0",
        env_cfg=stepstone_16dof_cfg(play=False),
