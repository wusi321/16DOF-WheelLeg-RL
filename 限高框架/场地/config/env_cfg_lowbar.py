"""限高框架（LowBar）训练配置 —— 2026-11-28 按官方规则重写。

**官方规格**（《2027 年足式机器人挑战赛 V1.0》表 1）：
  限高框架内高 290mm，内宽 800mm，长 800mm。限高框架颜色为绿色。

**通过定义**（表 2）：
  机器人以任何运动姿态从限高框架下穿过（允许触碰，但不允许暴力撞击）。

和旧版的区别（旧版是错的）：
  旧：一根圆柱横杆（半径 25mm）+ 两个支座，内高 0.25~0.35、宽 1.0m
  新：800×800 方形框架（4 柱 + 4 顶梁 + 中间加强梁），内高 0.29~0.36

放置位置：src/robot/config/env_cfgs.py（放在高墙之后、轮胎阵之前）
"""
LOWBAR_TERRAIN_SIZE = (8.0, 1.6)
LOWBAR_START_LENGTH = 0.8
LOWBAR_END_LENGTH = 0.8
LOWBAR_GOAL_DISTANCE = (
    LOWBAR_TERRAIN_SIZE[0] - LOWBAR_END_LENGTH - LOWBAR_START_LENGTH / 2.0
)
LOWBAR_NUM_LEVELS = 10
LOWBAR_EPISODE_S = 30.0


def lowbar_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """限高框架（官方规则表 1）——2026-11-28 按官方规格重写。

    **官方规格**：限高框架内高 290mm、内宽 800mm、长 800mm，颜色绿色。
    **通过定义**（表 2）：以任何运动姿态从框架下穿过，允许触碰但不允许暴力撞移。

    地形 `RCLowBarTerrainCfg`：一个 800×800 的方形框架（4 柱 + 顶梁 + 中间加强梁），
    难度 0 内高 0.36 m（好钻），难度 1 内高 0.29 m（官方标准，必须趴很低）。

    旧版做的是"一根圆柱横杆"，不是官方那个方框，已废弃。

    **判据为什么只写"前进 + 姿态"、不额外加"蹲低奖励"**：
    杠是真的碰撞体，站高了会被它挡住、走不过去，前进奖励自然上不来。
    物理约束本身就是最好的老师，硬加一个"趴低给分"反而可能让它趴到刮地。
    （项目里虽然有个 crouch_height_reward，但它绑定在"模式指令"系统上，
     这个场地没有那套指令，用不了。）
    """
    cfg = _make_base_env_cfg()
    cfg.scene.entities = {"robot": get_robot_cfg()}

    cfg.scene.terrain = TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            size=LOWBAR_TERRAIN_SIZE,
            border_width=2.0,
            num_rows=LOWBAR_NUM_LEVELS,
            num_cols=1,
            curriculum=True,
            difficulty_range=(0.0, 1.0),
            sub_terrains={
                "low_bar": RCLowBarTerrainCfg(
                    proportion=1.0,
                    size=LOWBAR_TERRAIN_SIZE,
                    clearance_range=(0.29, 0.36),
                    inner_width=0.80,
                    inner_length=0.80,
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
        twist.ranges.lin_vel_x = (0.3, 0.7)
        twist.ranges.lin_vel_y = (0.0, 0.0)
        twist.ranges.ang_vel_z = (0.0, 0.0)

    cfg.rewards.update(
        {
            "forward_progress": RewardTermCfg(
                func=forward_progress,
                weight=1.8,
                params={"command_name": "twist", "target_speed": 0.7},
            ),
            "track_lin_vel_x": RewardTermCfg(
                func=track_linear_velocity_x,
                weight=0.6,
                params={"std": 0.25, "command_name": "twist"},
            ),
            "goal_reached": RewardTermCfg(
                func=goal_reached_bonus, weight=80.0, params={"distance": LOWBAR_GOAL_DISTANCE}
            ),
            "is_terminated": RewardTermCfg(func=envs_mdp.is_terminated, weight=-10.0),
            "time_out": RewardTermCfg(
                func=time_out_penalty,
                weight=1.0,
                params={"distance": LOWBAR_GOAL_DISTANCE, "scale": -10.0},
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
            func=reached_goal, params={"distance": LOWBAR_GOAL_DISTANCE}
        ),
        "nan_detection": TerminationTermCfg(func=envs_mdp.nan_detection),
    }

    cfg.curriculum = {
        "terrain_levels": CurriculumTermCfg(
            func=stepstone_levels,
            params={"goal_distance": LOWBAR_GOAL_DISTANCE, "min_progress_frac": 0.6},
        ),
    }

    cfg.episode_length_s = LOWBAR_EPISODE_S
    cfg.decimation = 4
    cfg.seed = 42
    cfg.sim = _simulation_cfg(ccd_iterations=16)

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.curriculum = {}
        if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
            max_level = int(os.environ.get("LOWBAR_PLAY_MAX_LEVEL", "9"))
            cfg.scene.terrain.terrain_generator.curriculum = True
            cfg.scene.terrain.terrain_generator.num_rows = max(1, max_level + 1)
            cfg.scene.terrain.terrain_generator.num_cols = int(os.environ.get("LOWBAR_PLAY_COLS", "12"))
            cfg.scene.terrain.max_init_terrain_level = max_level

    return cfg
