"""平地「站姿 / 直行 / 步态转向」三阶段速度跟踪任务配置。

任务 ID（在 ``robot/__init__.py`` 注册）：

* ``Robot-WalkTurn-Stand-v0``    —— 零指令站姿（Step 2 门禁）
* ``Robot-WalkTurn-Straight-v0`` —— 只给前进 vx（Step 3 直行）
* ``Robot-WalkTurn-v0``          —— 直行 + 原地转向 + 弧线（Step 4 转向）

设计要点
--------
* 地形只用平地（``BoxFlatTerrainCfg``），观测轻。
* 指令用 ``StraightTurnVelocityCommand``（``mdp/commands.py``），按模式分类采样。
* 三个阶段的区别只在：指令范围/模式权重 + 奖励权重。观测、动作、执行器完全一致。
* 站姿门禁的核心：``base_height_l2`` 权重从旧 v2 的 -2 提到 -30/-20，
  并加 ``joint_deviation_l2``（站立）/``joint_pos_penalty``（行走），
  防止策略像旧 v2 一样把腿伸直、站到 0.47 m。
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import action_rate_l2, joint_pos_limits, joint_torques_l2
from mjlab.envs.mdp import dr as envs_dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.terrains import BoxFlatTerrainCfg, TerrainEntityCfg, TerrainGeneratorCfg
from mjlab.utils import spec_config as spec_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from .. import naming
from ..mdp.commands import StraightTurnVelocityCommandCfg
from ..mdp.lowpass_actions import (
    JointPositionDelayedLowPassActionCfg,
    JointVelocityDelayedLowPassActionCfg,
)
from ..mdp.rewards import (
    base_height_l2,
    contact_fraction_reward,
    flat_orientation_l2,
    joint_deviation_l2,
    joint_pos_penalty,
    lin_vel_y_l2,
    lin_vel_z_l2,
    safe_base_lin_vel,
    safe_foot_contact,
    stand_still_xy_l2,
    track_angular_velocity_z,
    track_linear_velocity_x,
    track_linear_velocity_y,
    wheel_roll_tracking,
)
from ..robot_cfg import LEG_POS_SCALE, WHEEL_VEL_SCALE, get_robot_cfg

# ===========================================================================
# 任务常量
# ===========================================================================
STAGE_STAND = "stand"
STAGE_STRAIGHT = "straight"
STAGE_TURN = "turn"
STAGE_TURN_RIGHT = "turn_right"
STAGE_TURN_LOCKED = "turn_locked"
STAGES = (STAGE_STAND, STAGE_STRAIGHT, STAGE_TURN, STAGE_TURN_RIGHT, STAGE_TURN_LOCKED)

FLAT_TERRAIN_SIZE = (8.0, 8.0)
EPISODE_LENGTH_S = 20.0
NUM_ENVS = 2048
ENV_SPACING = 2.5

#: 速度指令范围（按阶段覆盖）。
LIN_VEL_X_RANGE = (0.2, 0.8)  # 直行/弧线只向前
LIN_VEL_Y_RANGE = (-0.3, 0.3)
ANG_VEL_Z_RANGE = (-0.8, 0.8)


def _simulation_cfg():
    from mjlab.sim import MujocoCfg, SimulationCfg

    return SimulationCfg(
        contact_sensor_maxmatch=128,
        nconmax=128,
        njmax=256,
        mujoco=MujocoCfg(
            timestep=0.005,
            impratio=100,
            cone="elliptic",
            ccd_iterations=16,
        ),
    )


def _make_sensors() -> tuple[ContactSensorCfg, ...]:
    """三个接触传感器：轮-地、机身-地、腿-地。"""
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="body", pattern=naming.WHEEL_BODY_PATTERNS, entity="robot"
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=3,
        track_air_time=True,
    )
    base_ground_cfg = ContactSensorCfg(
        name="base_ground_contact",
        primary=ContactMatch(
            mode="body", pattern=rf"^{naming.BASE_BODY}$", entity="robot"
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="none",
        num_slots=1,
        history_length=4,
    )
    body_collision_cfg = ContactSensorCfg(
        name="body_collision",
        primary=ContactMatch(
            mode="body", pattern=naming.BODY_COLLISION_PATTERNS, entity="robot"
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )
    return (feet_ground_cfg, base_ground_cfg, body_collision_cfg)


def _make_observations() -> dict[str, ObservationGroupCfg]:
    """actor 只用本体感知；critic 额外看线速度和轮子接触（特权信息）。"""
    leg_cfg = SceneEntityCfg("robot", joint_names=naming.LEG_JOINT_PATTERNS)
    wheel_cfg = SceneEntityCfg("robot", joint_names=naming.WHEEL_JOINT_PATTERNS)

    actor_terms = {
        "base_ang_vel": ObservationTermCfg(
            func=envs_mdp.base_ang_vel, scale=0.25, noise=Unoise(n_min=-0.2, n_max=0.2)
        ),
        "projected_gravity": ObservationTermCfg(
            func=velocity_mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)
        ),
        "command": ObservationTermCfg(
            func=velocity_mdp.generated_commands, params={"command_name": "twist"}
        ),
        "joint_pos": ObservationTermCfg(
            func=envs_mdp.joint_pos_rel,
            params={"asset_cfg": leg_cfg},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "joint_vel": ObservationTermCfg(
            func=envs_mdp.joint_vel_rel,
            params={"asset_cfg": leg_cfg},
            scale=0.05,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        ),
        "wheel_vel": ObservationTermCfg(
            func=envs_mdp.joint_vel_rel,
            params={"asset_cfg": wheel_cfg},
            scale=0.05,
            noise=Unoise(n_min=-1.0, n_max=1.0),
        ),
        "actions": ObservationTermCfg(func=velocity_mdp.last_action),
    }

    critic_terms = {
        **actor_terms,
        "base_lin_vel": ObservationTermCfg(func=safe_base_lin_vel, scale=2.0),
        "foot_contact": ObservationTermCfg(
            func=safe_foot_contact, params={"sensor_name": "feet_ground_contact"}
        ),
    }

    return {
        "actor": ObservationGroupCfg(
            terms=actor_terms, concatenate_terms=True, enable_corruption=True
        ),
        "critic": ObservationGroupCfg(
            terms=critic_terms, concatenate_terms=True, enable_corruption=False
        ),
    }


def _make_actions(stage: str) -> dict:
    """12 个腿关节位置 + 低通 + 随机延迟；4 个轮子速度 + 低通。

    ``turn_locked`` 阶段把轮子动作 scale 设为 0：策略仍然输出 16 维，
    但 4 个轮子电机的目标恒为 0，相当于电机刹车，只靠腿完成转向。
    """
    wheel_scale = 0.0 if stage == STAGE_TURN_LOCKED else WHEEL_VEL_SCALE
    return {
        "leg_joint_pos": JointPositionDelayedLowPassActionCfg(
            entity_name="robot",
            actuator_names=naming.LEG_JOINT_PATTERNS,
            scale={".*_hip_joint": LEG_POS_SCALE * 0.5, "^(?!.*_hip_joint).*": LEG_POS_SCALE},
            use_default_offset=True,
            control_frequency=50.0,
            cut_off_frequency=5.0,
            min_delay=0,
            max_delay=2,
        ),
        "wheel_joint_vel": JointVelocityDelayedLowPassActionCfg(
            entity_name="robot",
            actuator_names=naming.WHEEL_JOINT_PATTERNS,
            scale=wheel_scale,
            offset=0.0,
            use_default_offset=False,
            control_frequency=50.0,
            cut_off_frequency=15.0,
            min_delay=0,
            max_delay=2,
        ),
    }


def _make_commands(stage: str) -> dict:
    """按阶段采样指令：站立全零 / 只直行 / 直行+转向+弧线。"""
    if stage == STAGE_STAND:
        cfg = StraightTurnVelocityCommandCfg(
            entity_name="robot",
            resampling_time_range=(10.0, 10.0),
            heading_command=False,
            rel_standing_envs=1.0,
            rel_heading_envs=0.0,
            rel_world_envs=0.0,
            rel_forward_envs=0.0,
            ranges=StraightTurnVelocityCommandCfg.Ranges(
                lin_vel_x=(0.0, 0.0),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(0.0, 0.0),
                heading=None,
            ),
        )
    elif stage == STAGE_STRAIGHT:
        cfg = StraightTurnVelocityCommandCfg(
            entity_name="robot",
            resampling_time_range=(10.0, 10.0),
            heading_command=False,
            rel_standing_envs=0.15,
            rel_heading_envs=0.0,
            rel_world_envs=0.0,
            rel_forward_envs=0.0,
            ranges=StraightTurnVelocityCommandCfg.Ranges(
                lin_vel_x=LIN_VEL_X_RANGE,
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(0.0, 0.0),
                heading=None,
            ),
            fraction_straight=1.0,
            fraction_turn_in_place=0.0,
            fraction_arc=0.0,
            fraction_lateral=0.0,
        )
    elif stage == STAGE_TURN:
        cfg = StraightTurnVelocityCommandCfg(
            entity_name="robot",
            resampling_time_range=(10.0, 10.0),
            heading_command=False,
            rel_standing_envs=0.10,
            rel_heading_envs=0.0,
            rel_world_envs=0.0,
            rel_forward_envs=0.0,
            ranges=StraightTurnVelocityCommandCfg.Ranges(
                lin_vel_x=LIN_VEL_X_RANGE,
                lin_vel_y=LIN_VEL_Y_RANGE,
                ang_vel_z=ANG_VEL_Z_RANGE,
                heading=None,
            ),
            fraction_straight=0.15,
            fraction_turn_in_place=0.45,
            fraction_arc=0.30,
            fraction_lateral=0.10,
        )
    elif stage == STAGE_TURN_RIGHT:
        # 只练右转（wz 为负），用来打破「只会左转、右转站着不动」的局部最优。
        cfg = StraightTurnVelocityCommandCfg(
            entity_name="robot",
            resampling_time_range=(10.0, 10.0),
            heading_command=False,
            rel_standing_envs=0.05,
            rel_heading_envs=0.0,
            rel_world_envs=0.0,
            rel_forward_envs=0.0,
            ranges=StraightTurnVelocityCommandCfg.Ranges(
                lin_vel_x=(0.0, 0.0),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(-0.8, -0.8),
                heading=None,
            ),
            fraction_straight=0.0,
            fraction_turn_in_place=1.0,
            fraction_arc=0.0,
            fraction_lateral=0.0,
        )
    elif stage == STAGE_TURN_LOCKED:
        # 轮子电机目标恒为 0（见 _make_actions），只做原地转向。
        cfg = StraightTurnVelocityCommandCfg(
            entity_name="robot",
            resampling_time_range=(10.0, 10.0),
            heading_command=False,
            rel_standing_envs=0.05,
            rel_heading_envs=0.0,
            rel_world_envs=0.0,
            rel_forward_envs=0.0,
            ranges=StraightTurnVelocityCommandCfg.Ranges(
                lin_vel_x=(0.0, 0.0),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=ANG_VEL_Z_RANGE,
                heading=None,
            ),
            fraction_straight=0.0,
            fraction_turn_in_place=1.0,
            fraction_arc=0.0,
            fraction_lateral=0.0,
        )
    else:
        raise ValueError(f"unknown stage: {stage!r} (expected one of {STAGES})")
    return {"twist": cfg}


def _make_events() -> dict:
    return {
        "reset_scene": EventTermCfg(func=envs_mdp.reset_scene_to_default, mode="reset"),
        "reset_base": EventTermCfg(
            func=envs_mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (-0.05, 0.05),
                    "y": (-0.05, 0.05),
                    "z": (0.0, 0.0),
                    "yaw": (-0.15, 0.15),
                },
                "velocity_range": {"x": (-0.1, 0.1), "y": (-0.05, 0.05), "z": (0.0, 0.0)},
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        "push_robot": EventTermCfg(
            func=envs_mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(6.0, 10.0),
            params={
                "velocity_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        # 轮子摩擦系数随机化：模拟不同地面（瓷砖/地毯/水泥）。
        "wheel_friction": EventTermCfg(
            func=envs_dr.geom_friction,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=(".*_wheel_Link_collision",)),
                "operation": "scale",
                "ranges": (0.8, 1.2),
            },
        ),
    }


def _common_rewards() -> dict:
    """三个阶段共用的姿态/平滑/能耗项。"""
    return {
        "upright": RewardTermCfg(
            func=velocity_mdp.upright,
            weight=1.0,
            params={
                "std": 0.5,
                "asset_cfg": SceneEntityCfg("robot", body_names=(naming.BASE_BODY,)),
            },
        ),
        "flat_orientation": RewardTermCfg(func=flat_orientation_l2, weight=-0.5),
        "lin_vel_z": RewardTermCfg(func=lin_vel_z_l2, weight=-0.1),
        "body_ang_vel": RewardTermCfg(
            func=velocity_mdp.body_angular_velocity_penalty,
            weight=-0.1,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=(naming.BASE_BODY,))
            },
        ),
        "joint_torques": RewardTermCfg(func=joint_torques_l2, weight=-2.0e-4),
        "joint_acc": RewardTermCfg(func=envs_mdp.joint_acc_l2, weight=-2.5e-7),
        "action_rate": RewardTermCfg(func=action_rate_l2, weight=-0.01),
        "joint_pos_limits": RewardTermCfg(func=joint_pos_limits, weight=-10.0),
        "body_collision": RewardTermCfg(
            func=velocity_mdp.self_collision_cost,
            weight=-1.0,
            params={"sensor_name": "body_collision"},
        ),
        "is_terminated": RewardTermCfg(func=envs_mdp.is_terminated, weight=-200.0),
    }


def _make_rewards(stage: str) -> dict:
    rewards = _common_rewards()
    if stage == STAGE_STAND:
        rewards.update(
            {
                # 站姿门禁：高度和关节角都要贴着 45° 标称姿态。
                # 站立时指令恒为 0；显式奖励“零速度”才能压住轮子缓漂。
                "track_lin_vel_x": RewardTermCfg(
                    func=track_linear_velocity_x,
                    weight=1.0,
                    params={"std": 0.10, "command_name": "twist"},
                ),
                "track_lin_vel_y": RewardTermCfg(
                    func=track_linear_velocity_y,
                    weight=1.0,
                    params={"std": 0.10, "command_name": "twist"},
                ),
                "track_ang_vel_z": RewardTermCfg(
                    func=track_angular_velocity_z,
                    weight=0.5,
                    params={"std": 0.15, "command_name": "twist"},
                ),
                "base_height_l2": RewardTermCfg(
                    func=base_height_l2,
                    weight=-30.0,
                    params={"target_height": naming.STAND_TARGET_HEIGHT},
                ),
                "joint_deviation_l2": RewardTermCfg(
                    func=joint_deviation_l2,
                    weight=-2.0,
                    params={"asset_cfg": SceneEntityCfg("robot", joint_names=naming.LEG_JOINT_PATTERNS)},
                ),
                "wheel_contact": RewardTermCfg(
                    func=contact_fraction_reward,
                    weight=0.2,
                    params={"sensor_name": "feet_ground_contact"},
                ),
            }
        )
    elif stage == STAGE_STRAIGHT:
        rewards.update(
            {
                "track_lin_vel_x": RewardTermCfg(
                    func=track_linear_velocity_x,
                    weight=1.0,
                    params={"std": 0.25, "command_name": "twist"},
                ),
                # 直行阶段必须把“零侧向速度 / 零偏航角速度”压住，
                # 否则策略会找一个“原地打转也能活”的局部最优。
                "track_lin_vel_y": RewardTermCfg(
                    func=track_linear_velocity_y,
                    weight=0.4,
                    params={"std": 0.25, "command_name": "twist"},
                ),
                "track_ang_vel_z": RewardTermCfg(
                    func=track_angular_velocity_z,
                    weight=1.0,
                    params={"std": 0.25, "command_name": "twist"},
                ),
                "base_height_l2": RewardTermCfg(
                    func=base_height_l2,
                    weight=-20.0,
                    params={"target_height": naming.STAND_TARGET_HEIGHT},
                ),
                "wheel_roll_tracking": RewardTermCfg(
                    func=wheel_roll_tracking,
                    weight=0.3,
                    params={
                        "command_name": "twist",
                        "wheel_radius": naming.WHEEL_RADIUS,
                        "wheel_track": naming.WHEEL_TRACK,
                        "side_signs": naming.WHEEL_SIDE_SIGNS,
                        "std": 3.0,
                        "asset_cfg": SceneEntityCfg("robot", joint_names=naming.WHEEL_JOINT_PATTERNS),
                    },
                ),
                "wheel_contact": RewardTermCfg(
                    func=contact_fraction_reward,
                    weight=0.3,
                    params={"sensor_name": "feet_ground_contact"},
                ),
                "feet_air_time": RewardTermCfg(
                    func=velocity_mdp.feet_air_time,
                    weight=0.8,
                    params={
                        "sensor_name": "feet_ground_contact",
                        "threshold_min": 0.1,
                        "threshold_max": 0.5,
                        "command_name": "twist",
                        "command_threshold": 0.1,
                    },
                ),
                "joint_pos_penalty": RewardTermCfg(
                    func=joint_pos_penalty,
                    weight=-0.2,
                    params={
                        "command_name": "twist",
                        "stand_still_scale": 2.0,
                        "velocity_threshold": 0.1,
                        "command_threshold": 0.1,
                        "asset_cfg": SceneEntityCfg("robot", joint_names=naming.LEG_JOINT_PATTERNS),
                    },
                ),
            }
        )
    elif stage in (STAGE_TURN, STAGE_TURN_RIGHT, STAGE_TURN_LOCKED):
        rewards.update(
            {
                "track_ang_vel_z": RewardTermCfg(
                    func=track_angular_velocity_z,
                    weight=(
                        4.0
                        if stage == STAGE_TURN_RIGHT
                        else (3.0 if stage == STAGE_TURN_LOCKED else 2.0)
                    ),
                    params={"std": 0.25, "command_name": "twist"},
                ),
                "track_lin_vel_x": RewardTermCfg(
                    func=track_linear_velocity_x,
                    weight=0.3 if stage == STAGE_TURN_RIGHT else 0.6,  # locked 也用 0.6
                    params={"std": 0.25, "command_name": "twist"},
                ),
                "track_lin_vel_y": RewardTermCfg(
                    func=track_linear_velocity_y,
                    weight=0.6 if stage == STAGE_TURN_RIGHT else 1.5,
                    params={"std": 0.25, "command_name": "twist"},
                ),
                "base_height_l2": RewardTermCfg(
                    func=base_height_l2,
                    weight=-40.0,
                    params={"target_height": naming.STAND_TARGET_HEIGHT},
                ),
                "lin_vel_y_l2": RewardTermCfg(
                    func=lin_vel_y_l2,
                    weight=-4.0,
                    params={},
                ),
                "stand_still_xy_l2": RewardTermCfg(
                    func=stand_still_xy_l2,
                    weight=-40.0,
                    params={"command_name": "twist", "threshold": 0.1},
                ),
                "wheel_roll_tracking": RewardTermCfg(
                    func=wheel_roll_tracking,
                    weight=0.4,
                    params={
                        "command_name": "twist",
                        "wheel_radius": naming.WHEEL_RADIUS,
                        "wheel_track": naming.WHEEL_TRACK,
                        "side_signs": naming.WHEEL_SIDE_SIGNS,
                        "std": 3.0,
                        "asset_cfg": SceneEntityCfg("robot", joint_names=naming.WHEEL_JOINT_PATTERNS),
                    },
                ),
                "wheel_contact": RewardTermCfg(
                    func=contact_fraction_reward,
                    weight=0.5,
                    params={"sensor_name": "feet_ground_contact"},
                ),
                "feet_air_time": RewardTermCfg(
                    func=velocity_mdp.feet_air_time,
                    weight=1.0,
                    params={
                        "sensor_name": "feet_ground_contact",
                        "threshold_min": 0.1,
                        "threshold_max": 0.5,
                        "command_name": "twist",
                        "command_threshold": 0.1,
                    },
                ),
                "joint_pos_penalty": RewardTermCfg(
                    func=joint_pos_penalty,
                    weight=-5.0,
                    params={
                        "command_name": "twist",
                        "stand_still_scale": 2.0,
                        "velocity_threshold": 0.1,
                        "command_threshold": 0.1,
                        "asset_cfg": SceneEntityCfg("robot", joint_names=naming.LEG_JOINT_PATTERNS),
                    },
                ),
                # 风格微调项：只覆盖 turn 阶段的同名奖励，stand/straight 不变。
                "action_rate": RewardTermCfg(func=action_rate_l2, weight=-0.02),
                "joint_acc": RewardTermCfg(func=envs_mdp.joint_acc_l2, weight=-5.0e-7),
                "flat_orientation": RewardTermCfg(func=flat_orientation_l2, weight=-0.8),
            }
        )
        if stage == STAGE_TURN_LOCKED:
            # 轮子被锁死，不应再奖励“按指令滚轮”。
            rewards.pop("wheel_roll_tracking", None)
    else:
        raise ValueError(f"unknown stage: {stage!r} (expected one of {STAGES})")
    return rewards


def _make_terminations() -> dict:
    return {
        "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
        "bad_orientation": TerminationTermCfg(
            func=envs_mdp.bad_orientation, params={"limit_angle": 1.0}
        ),
        "base_ground_contact": TerminationTermCfg(
            func=velocity_mdp.illegal_contact,
            params={"sensor_name": "base_ground_contact"},
        ),
        "nan_detection": TerminationTermCfg(func=envs_mdp.nan_detection),
    }


# 黑底白线网格地面：MuJoCo 原生 viewer 里比默认灰白平地对比度高，
# 也方便看轮子/脚端是否真的在动。纹理只影响外观，不影响物理。
_GROUND_GRID_TEXTURE = spec_cfg.TextureCfg(
    name="groundplane",
    type="2d",
    builtin="checker",
    mark="edge",
    rgb1=(0.012, 0.012, 0.012),
    rgb2=(0.012, 0.012, 0.012),
    markrgb=(0.88, 0.88, 0.88),
    width=512,
    height=512,
)
_GROUND_GRID_MATERIAL = spec_cfg.MaterialCfg(
    name="groundplane",
    texuniform=True,
    texrepeat=(8.0, 8.0),  # 8 m 地块 -> 1 m 一格
    reflectance=0.0,
    texture="groundplane",
    geom_names_expr=("terrain_.*",),
)


def walk_turn_env_cfg(stage: str = STAGE_TURN, play: bool = False) -> ManagerBasedRlEnvCfg:
    """三阶段共用环境；``stage`` 决定指令和奖励。"""
    if stage not in STAGES:
        raise ValueError(f"unknown stage: {stage!r} (expected one of {STAGES})")

    terrain = TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            size=FLAT_TERRAIN_SIZE,
            border_width=20.0,
            num_rows=10,
            num_cols=20,
            curriculum=False,
            sub_terrains={"flat": BoxFlatTerrainCfg(proportion=1.0)},
        ),
        textures=(_GROUND_GRID_TEXTURE,),
        materials=(_GROUND_GRID_MATERIAL,),
    )
    # 回放/演示只需要一个机器人；训练才用 2048 并行。
    num_envs = 1 if play else NUM_ENVS
    terrain.num_envs = num_envs
    terrain.env_spacing = ENV_SPACING

    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            num_envs=num_envs,
            env_spacing=ENV_SPACING,
            entities={"robot": get_robot_cfg()},
            terrain=terrain,
            sensors=_make_sensors(),
        ),
        commands=_make_commands(stage),
        actions=_make_actions(stage),
        observations=_make_observations(),
        rewards=_make_rewards(stage),
        terminations=_make_terminations(),
        events=_make_events(),
        metrics={"mean_leg_action_acc": MetricsTermCfg(func=velocity_mdp.mean_action_acc)},
        curriculum={},
        decimation=4,
        episode_length_s=EPISODE_LENGTH_S,
        sim=_simulation_cfg(),
        viewer=ViewerConfig(
            body_name=naming.BASE_BODY, distance=3.0, elevation=-20.0, azimuth=45.0
        ),
    )
    cfg.only_positive_rewards = False

    if play:
        cfg.episode_length_s = 1e9
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.events.pop("wheel_friction", None)

    return cfg