"""砖木坑（BrickPit）训练配置 —— 2026-11-28 按官方规则新建。

**官方规格**（《2027 年足式机器人挑战赛 V1.0》表 1）：
  L 形砖木坑 2000×2000mm，短边宽 1000mm、长边 2000mm；
  护栏高 150mm、绿色；护栏内分两区，分别填碎砖头与腐熟松树皮（10-20mm）。

**通过定义**（表 2）：
  必须从短边进入或离开；跨越过程中 1/2 以上足端与坑外地面接触 → 越障失败。

放置位置：src/robot/config/env_cfgs.py
"""
BRICKPIT_TERRAIN_SIZE = (4.0, 3.0)
BRICKPIT_GOAL_DISTANCE = 2.0
BRICKPIT_NUM_LEVELS = 10
BRICKPIT_EPISODE_S = 30.0


def brickpit_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """砖木坑（官方规则表 1，**L 形**）——2026-11-28 按官方规格新建。

    **官方规格**（《2027 年足式机器人挑战赛 V1.0》表 1）：
      L 形砖木坑 2000×2000mm，短边宽 1000mm、长边 2000mm；
      护栏高 150mm、绿色；护栏内分两区，分别填碎砖头与腐熟松树皮（10-20mm）。

    **通过定义**（表 2）：
      必须从**短边**进入或离开；跨越过程中若 1/2 以上足端与**坑外地面**接触 → 越障失败。

    实现要点：
    - 六段护栏围成 L 形闭环（机器人需要越过 150mm 护栏进出）
    - 坑内铺随机碎块模拟松散材料（低摩擦 0.45），难度越高越稀疏、越松
    - 速度指令压低到 0.2~0.5：坑里走不快，给高速度只会教出"猛冲打滑"
    """
    cfg = _make_base_env_cfg()
    cfg.scene.entities = {"robot": get_robot_cfg()}

    cfg.scene.terrain = TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            size=BRICKPIT_TERRAIN_SIZE,
            border_width=2.0,
            num_rows=BRICKPIT_NUM_LEVELS,
            num_cols=1,
            curriculum=True,
            difficulty_range=(0.0, 1.0),
            sub_terrains={
                "brickpit": BoxBrickPitTerrainCfg(
                    proportion=1.0,
                    size=BRICKPIT_TERRAIN_SIZE,
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
        twist.ranges.lin_vel_x = (0.2, 0.5)
        twist.ranges.lin_vel_y = (0.0, 0.0)
        twist.ranges.ang_vel_z = (0.0, 0.0)

    cfg.rewards.update(
        {
            "forward_progress": RewardTermCfg(
                func=forward_progress,
                weight=2.0,
                params={"command_name": "twist", "target_speed": 0.4},
            ),
            "track_lin_vel_x": RewardTermCfg(
                func=track_linear_velocity_x,
                weight=0.6,
                params={"std": 0.25, "command_name": "twist"},
            ),
            "goal_reached": RewardTermCfg(
                func=goal_reached_bonus, weight=80.0, params={"distance": BRICKPIT_GOAL_DISTANCE}
            ),
            "is_terminated": RewardTermCfg(func=envs_mdp.is_terminated, weight=-10.0),
            "time_out": RewardTermCfg(
                func=time_out_penalty,
                weight=1.0,
                params={"distance": BRICKPIT_GOAL_DISTANCE, "scale": -10.0},
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
        "base_too_low": TerminationTermCfg(func=base_too_low, params={"min_height": 0.10}),
        "goal_reached": TerminationTermCfg(
            func=reached_goal, params={"distance": BRICKPIT_GOAL_DISTANCE}
        ),
        "nan_detection": TerminationTermCfg(func=envs_mdp.nan_detection),
    }

    cfg.curriculum = {
        "terrain_levels": CurriculumTermCfg(
            func=stepstone_levels,
            params={"goal_distance": BRICKPIT_GOAL_DISTANCE, "min_progress_frac": 0.6},
        ),
    }

    cfg.episode_length_s = BRICKPIT_EPISODE_S
    cfg.decimation = 4
    cfg.seed = 42
    cfg.sim = _simulation_cfg(ccd_iterations=16)

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.curriculum = {}
        if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
            max_level = int(os.environ.get("BRICKPIT_PLAY_MAX_LEVEL", "9"))
            cfg.scene.terrain.terrain_generator.curriculum = True
            cfg.scene.terrain.terrain_generator.num_rows = max(1, max_level + 1)
            cfg.scene.terrain.terrain_generator.num_cols = int(os.environ.get("BRICKPIT_PLAY_COLS", "12"))
            cfg.scene.terrain.max_init_terrain_level = max_level

    return cfg
