"""绕桩（Slalom）训练配置 —— 2026-11-28 按官方规则重写。

**官方规格**（《2027 年足式机器人挑战赛 V1.0》表 1）：
  竖桩布局为 L 形，每个竖桩直径 100mm、高 100mm，桩之间中心距 900mm；
  绕桩区两端及拐角外侧各有一个直径 300mm 的圆形必达区；竖桩绿色。

和旧版的区别（旧版是错的）：
  旧：方柱 50×50×300mm、一字排列、柱距 1.00→0.52m 递减、每根柱旁一个必经点
  新：圆柱 直径 100mm、高 100mm、L 形 6 根、中心距固定 900mm、只有 3 个必达区

放置位置：src/robot/config/env_cfgs.py（放在 gapcourse 之后、combo 之前）
"""
SLALOM_TERRAIN_SIZE = (6.0, 3.0)
SLALOM_START_LENGTH = 0.8
SLALOM_END_LENGTH = 0.8
# L 形路径：水平臂 2.7m + 竖直臂 1.8m，斜向直线约 4.4m。
# 取 4.2 作为"到达终点"的判定距离（沿 x 推进会先到拐角，再转向 +y）。
SLALOM_GOAL_DISTANCE = 4.2
SLALOM_NUM_LEVELS = 10
SLALOM_EPISODE_S = 30.0


def slalom_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """绕桩课程（官方规格，2026-11-28 按规则重写）。

    **官方规格**（《2027 年足式机器人挑战赛 V1.0》表 1）：
      竖桩布局为 L 形，每个竖桩直径 100mm、高 100mm，桩之间中心距 900mm；
      绕桩区两端及拐角外侧各有一个直径 300mm 的圆形必达区；竖桩绿色。

    训练要点：
    - 6 根桩（水平臂 4 根含拐角 + 竖直臂额外 2 根）
    - 3 个必达区，**必须按 A → C → B 顺序经过**（水平端 → 拐角外侧 → 竖直端）
    - 桩只有 10cm 高，轮子能压过去，所以难点是"绕过"不是"跨越"
    - 奖励：按顺序经过必达区 +5、全部走完 +30、跳点 −30、蹭桩 −5
    """
    cfg = _make_base_env_cfg()
    cfg.scene.entities = {"robot": get_robot_cfg()}

    cfg.scene.terrain = TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            size=SLALOM_TERRAIN_SIZE,
            border_width=2.0,
            num_rows=SLALOM_NUM_LEVELS,
            num_cols=1,
            curriculum=True,
            difficulty_range=(0.0, 1.0),
            sub_terrains={
                "slalom": BoxSlalomTerrainCfg(
                    proportion=1.0,
                    size=SLALOM_TERRAIN_SIZE,
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
                func=forward_progress, weight=1.2, params={"command_name": "twist", "target_speed": 0.8}
            ),
            "track_lin_vel_x": RewardTermCfg(
                func=track_linear_velocity_x, weight=0.5, params={"std": 0.25, "command_name": "twist"}
            ),
            "track_ang_vel_z": RewardTermCfg(
                func=track_angular_velocity_z, weight=0.3, params={"std": 0.4, "command_name": "twist"}
            ),
            "waypoint": RewardTermCfg(func=waypoint_reward, weight=5.0),
            "all_waypoints": RewardTermCfg(func=all_waypoints, weight=30.0),
            "skip": RewardTermCfg(func=skip_penalty, weight=-30.0),
            "pillar_hit": RewardTermCfg(func=pillar_hit_penalty, weight=-5.0),
            "goal_reached": RewardTermCfg(
                func=goal_reached_bonus, weight=60.0, params={"distance": SLALOM_GOAL_DISTANCE}
            ),
            "is_terminated": RewardTermCfg(func=envs_mdp.is_terminated, weight=-10.0),
            "time_out": RewardTermCfg(
                func=time_out_penalty,
                weight=1.0,
                params={"distance": SLALOM_GOAL_DISTANCE, "scale": -10.0},
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
            func=reached_goal, params={"distance": SLALOM_GOAL_DISTANCE}
        ),
        "nan_detection": TerminationTermCfg(func=envs_mdp.nan_detection),
    }

    cfg.metrics.update(
        {
            "slalom_progress": MetricsTermCfg(func=slalom_progress),
            "wheel_slip": MetricsTermCfg(
                func=wheel_slip,
                params={"sensor_name": "feet_ground_contact", "wheel_radius": naming.WHEEL_RADIUS},
            ),
        }
    )

    cfg.curriculum = {
        "terrain_levels": CurriculumTermCfg(
            func=stepstone_levels,
            params={"goal_distance": SLALOM_GOAL_DISTANCE, "min_progress_frac": 0.6},
        ),
    }

    cfg.episode_length_s = SLALOM_EPISODE_S
    cfg.decimation = 4
    cfg.seed = 42
    cfg.sim = _simulation_cfg(ccd_iterations=16)

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.curriculum = {}
        if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
            max_level = int(os.environ.get("SLALOM_PLAY_MAX_LEVEL", "9"))
            cfg.scene.terrain.terrain_generator.curriculum = True
            cfg.scene.terrain.terrain_generator.num_rows = max(1, max_level + 1)
            cfg.scene.terrain.terrain_generator.num_cols = int(os.environ.get("SLALOM_PLAY_COLS", "12"))
            cfg.scene.terrain.max_init_terrain_level = max_level

    return cfg
