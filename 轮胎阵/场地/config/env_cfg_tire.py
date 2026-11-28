"""轮胎阵（TireArray）训练配置 —— 摘自原项目 env_cfgs.py（2026-11-28 新写）。

放置位置：src/robot/config/env_cfgs.py（放在 lowbar_env_cfg 之后、stand_env_cfg 之前）
依赖：terrains/tire_array.py（BoxTireArrayTerrainCfg，需要同时在 terrains/__init__.py 导出）

规格全部来自赛事官方场地模型 Robocon2027.xml：
  单个轮胎 外径 0.56 / 内孔 0.34 / 高 0.165 m
  官方碰撞体 = 12 个半径 0.055 的圆柱摆在半径 0.235 的环上
  阵列 4 x 2 共 8 个，横纵中心距均 0.56 m（彼此相切）

任务名：Robot-16DOF-TireArray-v0（已在 task_16dof.py 注册）
"""
# 轮胎阵（TireArray）
# -------------------------------------------------------------------------------------
# 规格全部来自赛事官方场地模型 Robocon2027.xml，地形定义见 terrains/tire_array.py：
#   单个轮胎外径 0.56 / 内孔 0.34 / 高 0.165 m，官方用 12 个圆柱围成环做碰撞体；
#   阵列 4 x 2 共 8 个，横纵中心距均 0.56 m（彼此相切）。
# 难点：胎面是宽仅 0.11 m 的窄环，轮子必须踩在环上，中间是深 0.165 m 的孔。
# =====================================================================================

TIRE_TERRAIN_SIZE = (8.0, 1.6)
TIRE_START_LENGTH = 0.8
TIRE_END_LENGTH = 0.8
TIRE_GOAL_DISTANCE = (
    TIRE_TERRAIN_SIZE[0] - TIRE_END_LENGTH - TIRE_START_LENGTH / 2.0
)
TIRE_NUM_LEVELS = 10
TIRE_EPISODE_S = 30.0


def tire_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """轮胎阵：踩在平放轮胎的窄胎面上通过（4 × 2 共 8 个）。

    难度靠**轮胎高度** 0.06 → 0.165 m 递增：
    6 cm 时轮子（直径 10.8 cm）能直接滚上去，等于平地；
    16.5 cm 时必须抬腿踩上 0.11 m 宽的窄环，踩偏就陷进中间的孔。

    判据只写常规的"前进 + 速度 + 到终点"，不加落点奖励——原因和低杠一样：
    踩空会真的掉进孔里、走不动，物理约束本身就是最好的老师。
    """
    cfg = _make_base_env_cfg()
    cfg.scene.entities = {"robot": get_robot_cfg()}

    cfg.scene.terrain = TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            size=TIRE_TERRAIN_SIZE,
            border_width=2.0,
            num_rows=TIRE_NUM_LEVELS,
            num_cols=1,
            curriculum=True,
            difficulty_range=(0.0, 1.0),
            sub_terrains={
                "tires": BoxTireArrayTerrainCfg(
                    proportion=1.0,
                    size=TIRE_TERRAIN_SIZE,
                    start_length=TIRE_START_LENGTH,
                    num_x=4,
                    num_y=2,
                    pitch=0.56,
                    height_range=(0.06, 0.165),
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
        twist.ranges.lin_vel_x = (0.3, 0.8)
        twist.ranges.lin_vel_y = (0.0, 0.0)
        twist.ranges.ang_vel_z = (0.0, 0.0)

    cfg.rewards.update(
        {
            "forward_progress": RewardTermCfg(
                func=forward_progress,
                weight=1.8,
                params={"command_name": "twist", "target_speed": 0.8},
            ),
            "track_lin_vel_x": RewardTermCfg(
                func=track_linear_velocity_x,
                weight=0.6,
                params={"std": 0.25, "command_name": "twist"},
            ),
            "goal_reached": RewardTermCfg(
                func=goal_reached_bonus, weight=80.0, params={"distance": TIRE_GOAL_DISTANCE}
            ),
            "is_terminated": RewardTermCfg(func=envs_mdp.is_terminated, weight=-10.0),
            "time_out": RewardTermCfg(
                func=time_out_penalty,
                weight=1.0,
                params={"distance": TIRE_GOAL_DISTANCE, "scale": -10.0},
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
            func=reached_goal, params={"distance": TIRE_GOAL_DISTANCE}
        ),
        "nan_detection": TerminationTermCfg(func=envs_mdp.nan_detection),
    }

    cfg.curriculum = {
        "terrain_levels": CurriculumTermCfg(
            func=stepstone_levels,
            params={"goal_distance": TIRE_GOAL_DISTANCE, "min_progress_frac": 0.6},
        ),
    }

    cfg.episode_length_s = TIRE_EPISODE_S
    cfg.decimation = 4
    cfg.seed = 42
    cfg.sim = _simulation_cfg(ccd_iterations=16)

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.curriculum = {}
        if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
            max_level = int(os.environ.get("TIRE_PLAY_MAX_LEVEL", "9"))
            cfg.scene.terrain.terrain_generator.curriculum = True
            cfg.scene.terrain.terrain_generator.num_rows = max(1, max_level + 1)
            cfg.scene.terrain.terrain_generator.num_cols = int(os.environ.get("TIRE_PLAY_COLS", "12"))
            cfg.scene.terrain.max_init_terrain_level = max_level

    return cfg


