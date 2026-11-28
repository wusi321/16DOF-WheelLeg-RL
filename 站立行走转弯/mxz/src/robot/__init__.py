"""16DOF-WheelLeg 轮足机器狗：站姿 → 直行 → 步态转向三阶段任务。

注册方式与 mjlab 官方任务一致：venv 里 ``robot`` 包的 entry point
``mjlab.tasks = robot`` 会在 ``import mjlab`` 时自动 import 本包，从而把下面的任务
写进 ``mjlab.tasks.registry``。只要 ``PYTHONPATH`` 指向本工程的 ``src``，
``python scripts/train.py <task-id>`` 就能看到这些任务。

已注册任务
----------
* ``Robot-WalkTurn-Stand-v0``：零指令站姿门禁（Step 2）。
* ``Robot-WalkTurn-Straight-v0``：只给前进 vx 的直行（Step 3）。
* ``Robot-WalkTurn-v0``：直行 + 原地转向 + 弧线（Step 4）。
"""

from mjlab.tasks.registry import register_mjlab_task

from .config.env_cfgs import (
    STAGE_STAND,
    STAGE_STRAIGHT,
    STAGE_TURN,
    STAGE_TURN_LOCKED,
    STAGE_TURN_RIGHT,
    walk_turn_env_cfg,
)
from .config.rl_cfg import walk_turn_ppo_runner_cfg

_TASKS = (
    ("Robot-WalkTurn-Stand-v0", STAGE_STAND, "robot_walk_turn_stand"),
    ("Robot-WalkTurn-Straight-v0", STAGE_STRAIGHT, "robot_walk_turn_straight"),
    ("Robot-WalkTurn-v0", STAGE_TURN, "robot_walk_turn"),
    ("Robot-WalkTurn-Right-v0", STAGE_TURN_RIGHT, "robot_walk_turn_right"),
    ("Robot-WalkTurn-Locked-v0", STAGE_TURN_LOCKED, "robot_walk_turn_locked"),
)

for _task_id, _stage, _experiment in _TASKS:
    register_mjlab_task(
        task_id=_task_id,
        env_cfg=walk_turn_env_cfg(_stage),
        play_env_cfg=walk_turn_env_cfg(_stage, play=True),
        rl_cfg=walk_turn_ppo_runner_cfg(_experiment),
    )