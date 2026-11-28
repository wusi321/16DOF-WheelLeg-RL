"""单木桥（SingleBeam）训练配置 —— 摘自原项目 env_cfgs.py（2026-11-28 新写）。

放置位置：src/robot/config/env_cfgs.py（放在 tire_env_cfg 之后、stand_env_cfg 之前）
依赖：terrains/single_beam.py（BoxSingleBeamTerrainCfg，需要在 terrains/__init__.py 导出）

官方规格（《2027 年足式机器人挑战赛 V1.0》规则 3.1 节）：
    单木桥长 3000mm，宽 400mm，高 100mm，表面涂绿色水性漆。
通过判定（规则 4.2 节）：从一侧短边进入、另一端短边离开即算通过；
    中途从侧面摔下、或 1/2 以上足接触地面（含 1/2），需重新通过。

任务名：Robot-16DOF-SingleBeam-v0（已在 task_16dof.py 注册）
"""
# 单木桥（SingleBeam）
# -------------------------------------------------------------------------------------
# 官方规格（2027 足式机器人挑战赛 V1.0 规则 3.1 节）：
#   单木桥长 3000mm，宽 400mm，高 100mm，表面涂绿色水性漆。
# 通过判定（规则 4.2 节）：从一侧短边进入、另一端短边离开即算通过；
#   中途从侧面摔下、或 1/2 以上足接触地面（含 1/2），需重新通过。
#
# 这是全场最考验"持续走直线"的障碍——梅花桩可以踩一下就走，单木桥要在
# 400mm 宽的窄面上保持平衡走完 3 米，两侧没有任何护栏。
# =====================================================================================

BEAM_TERRAIN_SIZE = (8.0, 1.6)
BEAM_START_LENGTH = 0.8
BEAM_END_LENGTH = 0.8
BEAM_GOAL_DISTANCE = (
    BEAM_TERRAIN_SIZE[0] - BEAM_END_LENGTH - BEAM_START_LENGTH / 2.0
)
BEAM_NUM_LEVELS = 10
BEAM_EPISODE_S = 30.0


def beam_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """单木桥：在 400mm 宽的窄木面上走完 3 米。

    课程靠桥高（0.06 → 0.10 m）和桥宽（0.40 → 0.32 m）双向收紧：
    先让它在矮一点的桥上学会"爬上去并保持"，再把桥加高、收窄到比比赛还窄，
    比赛时的 400mm 就会有安全余量。

    判据只写"前进 + 速度 + 到终点"：掉下桥就是踩到地面，会触发
    `base_too_low`（阈值设得比桥面低一点点）和前进停滞，物理约束足够。
    """
    cfg = _make_base_env_cfg()
    cfg.scene.entities = {"robot": get_robot_cfg()}

    cfg.scene.terrain = TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            size=BEAM_TERRAIN_SIZE,
            border_width=2.0,
            num_rows=BEAM_NUM_LEVELS,
            num_cols=1,
            curriculum=True,
            difficulty_range=(0.0, 1.0),
            sub_terrains={
                "beam": BoxSingleBeamTerrainCfg(
                    proportion=1.0,
                    size=BEAM_TERRAIN_SIZE,
                    start_length=BEAM_START_LENGTH,
                    beam_length=3.0,
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
                weight=2.0,
                params={"command_name": "twist", "target_speed": 0.7},
            ),
            "track_lin_vel_x": RewardTermCfg(
                func=track_linear_velocity_x,
                weight=0.6,
                params={"std": 0.25, "command_name": "twist"},
            ),
            "goal_reached": RewardTermCfg(
                func=goal_reached_bonus, weight=80.0, params={"distance": BEAM_GOAL_DISTANCE}
            ),
            "is_terminated": RewardTermCfg(func=envs_mdp.is_terminated, weight=-10.0),
            "time_out": RewardTermCfg(
                func=time_out_penalty,
                weight=1.0,
                params={"distance": BEAM_GOAL_DISTANCE, "scale": -10.0},
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
            func=reached_goal, params={"distance": BEAM_GOAL_DISTANCE}
        ),
        "nan_detection": TerminationTermCfg(func=envs_mdp.nan_detection),
    }

    cfg.curriculum = {
        "terrain_levels": CurriculumTermCfg(
            func=stepstone_levels,
            params={"goal_distance": BEAM_GOAL_DISTANCE, "min_progress_frac": 0.6},
        ),
    }

    cfg.episode_length_s = BEAM_EPISODE_S
    cfg.decimation = 4
    cfg.seed = 42
    cfg.sim = _simulation_cfg(ccd_iterations=16)

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.curriculum = {}
        if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
            max_level = int(os.environ.get("BEAM_PLAY_MAX_LEVEL", "9"))
            cfg.scene.terrain.terrain_generator.curriculum = True
            cfg.scene.terrain.terrain_generator.num_rows = max(1, max_level + 1)
            cfg.scene.terrain.terrain_generator.num_cols = int(os.environ.get("BEAM_PLAY_COLS", "12"))
            cfg.scene.terrain.max_init_terrain_level = max_level

    return cfg


