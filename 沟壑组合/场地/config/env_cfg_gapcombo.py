"""沟壑组合（GapCourse）训练配置 —— 2026-11-28 按官方规则重写。

**官方规格**（《2027 年足式机器人挑战赛 V1.0》表 1）：
  L 形；两顶部平台高 360mm、800×800mm；台阶深 320mm、高 120mm、宽 800mm；
  两平台之间沟壑宽 360mm；斜坡坡度 20°；除启动区 2（黄色）外表面绿色。

**通过定义**（表 2）：
  可从任一方向迈上台阶/斜坡，通过沟壑后从另一方向的斜坡/台阶回到地面；
  每级台阶顶面至少与足底接触一次；1/2 以上足与地面接触 → 越障失败。

和旧版的区别（旧版是错的）：
  旧："一串 40cm 平台 + 越来越宽的沟（5→45cm）"，那不是官方障碍
  新：官方 L 形组合（台阶 + 平台 + 沟 + 斜坡）

放置位置：src/robot/config/env_cfgs.py
"""
GAPCOURSE_TERRAIN_SIZE = (5.2, 3.2)
GAPCOURSE_GOAL_DISTANCE = GAPCOURSE_TERRAIN_SIZE[0] - 0.80
GAPCOURSE_NUM_LEVELS = 10
GAPCOURSE_EPISODE_S = 25.0


def gapcourse_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """沟壑组合（官方规则表 1，**L 形**）——2026-11-28 按官方规格重写。

    **官方规格**（《2027 年足式机器人挑战赛 V1.0》表 1）：
      L 形；两顶部平台高 360mm、800×800mm；台阶深 320mm、高 120mm、宽 800mm；
      两平台之间沟壑宽 360mm；斜坡坡度 20°；除启动区 2（黄色）外表面绿色。

    **通过定义**（表 2）：
      可从任一方向迈上台阶/斜坡，通过沟壑后从另一方向的斜坡/台阶回到地面；
      **每级台阶顶面至少与足底接触一次**；1/2 以上足与地面接触 → 越障失败。

    几何（XML 实测）：
      沿 -x：台阶(顶0.12) → 台阶(顶0.24) → 平台A(顶0.36) → 沟(0.36) → 平台B(顶0.36)
      平台 B 的 +y 侧接 20° 斜坡下到地面  ← 这一拐就是"L 形"

    难度：只缩放高度与沟宽（官方尺寸为难度 1），台阶深度/平台边长保持官方值。
    """
    cfg = _make_base_env_cfg()
    cfg.scene.entities = {"robot": get_robot_cfg()}

    cfg.scene.terrain = TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            size=GAPCOURSE_TERRAIN_SIZE,
            border_width=2.0,
            num_rows=GAPCOURSE_NUM_LEVELS,
            num_cols=1,
            curriculum=True,
            difficulty_range=(0.0, 1.0),
            sub_terrains={
                "gapcombo": BoxGapCourseTerrainCfg(
                    proportion=1.0,
                    size=GAPCOURSE_TERRAIN_SIZE,
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
                func=forward_progress, weight=1.5, params={"command_name": "twist", "target_speed": 0.8}
            ),
            "track_lin_vel_x": RewardTermCfg(
                func=track_linear_velocity_x, weight=0.6, params={"std": 0.25, "command_name": "twist"}
            ),
            "track_ang_vel_z": RewardTermCfg(
                func=track_angular_velocity_z, weight=0.3, params={"std": 0.4, "command_name": "twist"}
            ),
            "goal_reached": RewardTermCfg(
                func=goal_reached_bonus, weight=80.0, params={"distance": GAPCOURSE_GOAL_DISTANCE}
            ),
            "is_terminated": RewardTermCfg(func=envs_mdp.is_terminated, weight=-10.0),
            "time_out": RewardTermCfg(
                func=time_out_penalty,
                weight=1.0,
                params={"distance": GAPCOURSE_GOAL_DISTANCE, "scale": -10.0},
            ),
            # 跃沟：离地 >=2 步后落地 +5（鼓励真的跳过去）
            "leap": RewardTermCfg(
                func=leap_bonus,
                weight=5.0,
                params={"min_air_steps": 2, "min_advance": 0.25},
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
            func=reached_goal, params={"distance": GAPCOURSE_GOAL_DISTANCE}
        ),
        "nan_detection": TerminationTermCfg(func=envs_mdp.nan_detection),
    }

    cfg.metrics.update(
        {
            "progress": MetricsTermCfg(
                func=forward_progress, params={"command_name": "twist", "target_speed": 0.8}
            ),
            "wheel_slip": MetricsTermCfg(
                func=wheel_slip,
                params={"sensor_name": "feet_ground_contact", "wheel_radius": naming.WHEEL_RADIUS},
            ),
            "gaps_crossed": MetricsTermCfg(func=gaps_crossed),
        }
    )

    cfg.curriculum = {
        "terrain_levels": CurriculumTermCfg(
            func=stepstone_levels,
            params={"goal_distance": GAPCOURSE_GOAL_DISTANCE, "min_progress_frac": 0.6},
        ),
    }

    cfg.episode_length_s = GAPCOURSE_EPISODE_S
    cfg.decimation = 4
    cfg.seed = 42
    cfg.sim = _simulation_cfg(ccd_iterations=16)

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.curriculum = {}
        if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
            max_level = int(os.environ.get("GAPCOURSE_PLAY_MAX_LEVEL", "9"))
            cfg.scene.terrain.terrain_generator.curriculum = True
            cfg.scene.terrain.terrain_generator.num_rows = max(1, max_level + 1)
            cfg.scene.terrain.terrain_generator.num_cols = int(os.environ.get("GAPCOURSE_PLAY_COLS", "12"))
            cfg.scene.terrain.max_init_terrain_level = max_level

    return cfg
