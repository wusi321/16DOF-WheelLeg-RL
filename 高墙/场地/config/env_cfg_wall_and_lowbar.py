"""高墙（Wall）/ 低杠（LowBar）训练配置 —— 摘自原项目 env_cfgs.py（2026-11-28 新写）。

放置位置：src/robot/config/env_cfgs.py（放在 combo_env_cfg 之后、stand_env_cfg 之前）

**这两个障碍原本没有独立训练配置**：
  - 高墙的地形只作为 rough_env_cfg 里占比 15% 的一种地形出现
  - 低杠的地形只作为 crawl_env_cfg 里的一种地形出现
  混在综合地形里练，策略会"挑软柿子捏"——粗糙地形里还有台阶、乱石、斜坡，
  它会绕开最难的那个。所以 2026-11-28 把它们拆成独立任务，专门练。

任务名：Robot-16DOF-Wall-v0 / Robot-16DOF-LowBar-v0（已在 task_16dof.py 注册）
"""
# =====================================================================================
# 高墙（Wall）与 低杠（LowBar）
# -------------------------------------------------------------------------------------
# 这两个障碍的地形其实早就写好了（competition_terrains.py），但一直只作为"配菜"混在
# 别的任务里：墙是 rough_env_cfg 里占比 15% 的一种地形，低杠是 crawl_env_cfg 里的一种。
# 混在一起练的问题是策略会"挑软柿子捏"——粗糙地形里还有台阶、乱石、斜坡，它会绕开
# 最难的那个。单独拎成一个任务，才能逼它专门练"翻墙"和"钻杠"。
# =====================================================================================

WALL_TERRAIN_SIZE = (8.0, 1.6)
WALL_START_LENGTH = 0.8
WALL_END_LENGTH = 0.8
WALL_GOAL_DISTANCE = (
    WALL_TERRAIN_SIZE[0] - WALL_END_LENGTH - WALL_START_LENGTH / 2.0
)
WALL_NUM_LEVELS = 10
WALL_EPISODE_S = 30.0


def wall_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """高墙：连续翻越 5 道横墙，练"抬腿蹬上去 + 越过去"。

    地形 `RCWallTerrainCfg`：墙 1.0 m 宽、5 cm 厚，高度随难度 0.0 → 0.35 m 递增。
    **难度 0 时墙高为 0（等于平地）**，这是刻意的——一上来就让轮腿机器人撞 20 cm 墙
    是学不会的，先让它在"没有墙"的赛道上把前进练顺，墙再从 0 一点点长起来。

    奖励里保留 `leap_bonus`（离地 >=2 步后落地给分），因为翻墙本质是一次小跳；
    同时不要加任何"离地/抬腿"惩罚，否则等于惩罚正确动作。
    """
    cfg = _make_base_env_cfg()
    cfg.scene.entities = {"robot": get_robot_cfg()}

    cfg.scene.terrain = TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            size=WALL_TERRAIN_SIZE,
            border_width=2.0,
            num_rows=WALL_NUM_LEVELS,
            num_cols=1,
            curriculum=True,
            difficulty_range=(0.0, 1.0),
            sub_terrains={
                "walls": RCWallTerrainCfg(
                    proportion=1.0,
                    size=WALL_TERRAIN_SIZE,
                    wall_height_range=(0.0, 0.35),
                    wall_centers_x=(2.1, 3.2, 4.3, 5.4, 6.5),
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
        twist.ranges.lin_vel_x = (0.4, 0.9)
        twist.ranges.lin_vel_y = (0.0, 0.0)
        twist.ranges.ang_vel_z = (0.0, 0.0)

    cfg.rewards.update(
        {
            "forward_progress": RewardTermCfg(
                func=forward_progress,
                weight=1.8,
                params={"command_name": "twist", "target_speed": 0.9},
            ),
            "track_lin_vel_x": RewardTermCfg(
                func=track_linear_velocity_x,
                weight=0.6,
                params={"std": 0.25, "command_name": "twist"},
            ),
            "goal_reached": RewardTermCfg(
                func=goal_reached_bonus, weight=80.0, params={"distance": WALL_GOAL_DISTANCE}
            ),
            "is_terminated": RewardTermCfg(func=envs_mdp.is_terminated, weight=-10.0),
            "time_out": RewardTermCfg(
                func=time_out_penalty,
                weight=1.0,
                params={"distance": WALL_GOAL_DISTANCE, "scale": -10.0},
            ),
            "leap": RewardTermCfg(
                func=leap_bonus,
                weight=4.0,
                params={"min_air_steps": 2, "min_advance": 0.20},
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
            func=reached_goal, params={"distance": WALL_GOAL_DISTANCE}
        ),
        "nan_detection": TerminationTermCfg(func=envs_mdp.nan_detection),
    }

    cfg.curriculum = {
        "terrain_levels": CurriculumTermCfg(
            func=stepstone_levels,
            params={"goal_distance": WALL_GOAL_DISTANCE, "min_progress_frac": 0.6},
        ),
    }

    cfg.episode_length_s = WALL_EPISODE_S
    cfg.decimation = 4
    cfg.seed = 42
    cfg.sim = _simulation_cfg(ccd_iterations=16)

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.curriculum = {}
        if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
            max_level = int(os.environ.get("WALL_PLAY_MAX_LEVEL", "9"))
            cfg.scene.terrain.terrain_generator.curriculum = True
            cfg.scene.terrain.terrain_generator.num_rows = max(1, max_level + 1)
            cfg.scene.terrain.terrain_generator.num_cols = int(os.environ.get("WALL_PLAY_COLS", "12"))
            cfg.scene.terrain.max_init_terrain_level = max_level

    return cfg


LOWBAR_TERRAIN_SIZE = (8.0, 1.6)
LOWBAR_START_LENGTH = 0.8
LOWBAR_END_LENGTH = 0.8
LOWBAR_GOAL_DISTANCE = (
    LOWBAR_TERRAIN_SIZE[0] - LOWBAR_END_LENGTH - LOWBAR_START_LENGTH / 2.0
)
LOWBAR_NUM_LEVELS = 10
LOWBAR_EPISODE_S = 30.0


def lowbar_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """低杠：机身必须趴低，钻过离地 0.25 → 0.35 m 的横杠。

    地形 `RCLowBarTerrainCfg`：一根沿横向的圆柱杠（半径 2.5 cm），
    难度 0 时离地最高 0.35 m（容易钻），难度 1 时最低 0.25 m（必须趴很低）。

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
                    clearance_range=(0.25, 0.35),
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


