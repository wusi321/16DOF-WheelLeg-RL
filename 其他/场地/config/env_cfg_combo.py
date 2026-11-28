"""组合场地（Combo）训练配置 —— 2026-11-28 同步官方规格。

两列地形并排：
  列1 = 沟壑组合（官方 L 形：平台 360mm、沟 360mm、台阶 320×120、20° 斜坡）
  列2 = 绕桩（官方 L 形：6 桩 ⌀100×100、中心距 900、3 个必达区 ⌀300）

**这个任务不是官方障碍，是自选的"混合训练"**：官方比赛里 8 个障碍是挨着摆的，
机器人要连续通过；用组合地形训练可以逼它"刚跨完沟就要绕桩"，更接近实赛节奏。

放置位置：src/robot/config/env_cfgs.py（放在 Slalom 之后、高墙之前）
"""
COMBO_TERRAIN_SIZE = (6.0, 3.2)
COMBO_START_LENGTH = 0.8
COMBO_END_LENGTH = 0.8
COMBO_GOAL_DISTANCE = (
    COMBO_TERRAIN_SIZE[0] - COMBO_END_LENGTH - COMBO_START_LENGTH / 2.0
)
COMBO_NUM_LEVELS = 10
COMBO_EPISODE_S = 28.0


def combo_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """两条赛道并排在一张地图里，一次训练同时练。

    mjlab 的 curriculum 模式给每个 sub-terrain 分配**一列**，环境按
    `terrain_types` 落到某一列 —— 于是同一个 scene 里同时存在沟壑列和绕桩列，
    各占一半环境（`proportion=0.5`）。

    奖励按赛道分发（`src/robot/mdp/combo.py` 里按 `terrain_types` 做掩码）：

    - 沟壑列：`leap`（跃沟 +5）、`gaps_crossed` 指标；
    - 绕桩列：`waypoint +5` / `all_waypoints +30` / `skip -30` / `pillar_hit -5`、
      `slalom_progress` 指标；
    - 两条赛道共用：前进、速度跟踪、姿态、终点、超时、摔倒。

    难度（沟宽 / 柱距）由同一个 `difficulty` 控制，10 档课程共用一套升级规则。
    """
    cfg = _make_base_env_cfg()
    cfg.scene.entities = {"robot": get_robot_cfg()}

    cfg.scene.terrain = TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            size=COMBO_TERRAIN_SIZE,
            border_width=2.0,
            num_rows=COMBO_NUM_LEVELS,
            num_cols=2,
            curriculum=True,
            difficulty_range=(0.0, 1.0),
            sub_terrains={
                "gapcombo": BoxGapCourseTerrainCfg(
                    proportion=0.5,
                    size=COMBO_TERRAIN_SIZE,
                ),
                "slalom": BoxSlalomTerrainCfg(
                    proportion=0.5,
                    size=COMBO_TERRAIN_SIZE,
                ),
            },
        ),
        max_init_terrain_level=0,
    )
    cfg.scene.terrain.num_envs = 2048
    cfg.scene.terrain.env_spacing = 2.5

    cfg.observations["actor"].terms["height_scan"] = ObservationTermCfg(
        func=safe_height_scan, params={"sensor_name": "height_scanner"}, clip=(-1.0, 1.0)
    )

    twist = cfg.commands["twist"]
    twist.rel_standing_envs = 0.0
    twist.rel_heading_envs = 0.0
    if twist.ranges is not None:
        twist.ranges.lin_vel_x = (0.4, 0.8)
        twist.ranges.lin_vel_y = (0.0, 0.0)
        twist.ranges.ang_vel_z = (0.0, 0.0)

    cfg.rewards.update(
        {
            "forward_progress": RewardTermCfg(
                func=forward_progress,
                weight=1.3,
                params={"command_name": "twist", "target_speed": 0.8},
            ),
            "track_lin_vel_x": RewardTermCfg(
                func=track_linear_velocity_x,
                weight=0.5,
                params={"std": 0.25, "command_name": "twist"},
            ),
            "track_ang_vel_z": RewardTermCfg(
                func=track_angular_velocity_z,
                weight=0.3,
                params={"std": 0.4, "command_name": "twist"},
            ),
            # ---- 沟壑列 ----
            "leap": RewardTermCfg(
                func=leap_bonus_on_gap,
                weight=5.0,
                params={"min_air_steps": 2, "min_advance": 0.25},
            ),
            # ---- 绕桩列 ----
            "waypoint": RewardTermCfg(func=waypoint_reward_on_slalom, weight=5.0),
            "all_waypoints": RewardTermCfg(func=all_waypoints_on_slalom, weight=30.0),
            "skip": RewardTermCfg(func=skip_penalty_on_slalom, weight=-30.0),
            "pillar_hit": RewardTermCfg(func=pillar_hit_on_slalom, weight=-5.0),
            # ---- 公共 ----
            "goal_reached": RewardTermCfg(
                func=goal_reached_bonus, weight=60.0, params={"distance": COMBO_GOAL_DISTANCE}
            ),
            "is_terminated": RewardTermCfg(func=envs_mdp.is_terminated, weight=-10.0),
            "time_out": RewardTermCfg(
                func=time_out_penalty,
                weight=1.0,
                params={"distance": COMBO_GOAL_DISTANCE, "scale": -10.0},
            ),
            "idle": RewardTermCfg(
                func=idle_penalty,
                weight=1.0,
                params={"idle_steps": 75, "min_gain": 0.03, "scale": -0.5},
            ),
            "overspeed": RewardTermCfg(
                func=overspeed, weight=-0.5, params={"command_name": "twist", "margin": 0.4}
            ),
        }
    )

    cfg.terminations = {
        "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
        "bad_orientation": TerminationTermCfg(
            func=envs_mdp.bad_orientation, params={"limit_angle": math.radians(60.0)}
        ),
        "base_ground_contact": TerminationTermCfg(
            func=velocity_mdp.illegal_contact, params={"sensor_name": "base_ground_contact"}
        ),
        "base_too_low": TerminationTermCfg(func=base_too_low, params={"min_height": 0.12}),
        "goal_reached": TerminationTermCfg(
            func=reached_goal, params={"distance": COMBO_GOAL_DISTANCE}
        ),
        "nan_detection": TerminationTermCfg(func=envs_mdp.nan_detection),
    }

    cfg.metrics.update(
        {
            "gaps_crossed": MetricsTermCfg(func=gaps_crossed_on_gap),
            "slalom_progress": MetricsTermCfg(func=slalom_progress_on_slalom),
            "wheel_slip": MetricsTermCfg(
                func=wheel_slip,
                params={"sensor_name": "feet_ground_contact", "wheel_radius": naming.WHEEL_RADIUS},
            ),
        }
    )

    cfg.curriculum = {
        "terrain_levels": CurriculumTermCfg(
            func=stepstone_levels,
            params={"goal_distance": COMBO_GOAL_DISTANCE, "min_progress_frac": 0.6},
        ),
    }

    cfg.episode_length_s = COMBO_EPISODE_S
    cfg.decimation = 4
    cfg.seed = 42
    cfg.sim = _simulation_cfg(ccd_iterations=16)

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.curriculum = {}
        if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
            max_level = int(os.environ.get("COMBO_PLAY_MAX_LEVEL", "9"))
            cfg.scene.terrain.terrain_generator.curriculum = True
            cfg.scene.terrain.terrain_generator.num_rows = max(1, max_level + 1)
            cfg.scene.terrain.max_init_terrain_level = max_level

    return cfg
