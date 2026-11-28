"""梅花桩（StepStone）训练配置 —— 摘自原项目 src/robot/config/env_cfgs.py 第 972-1442 行。

文件名：env_cfgs.py 里的 stepstone_env_cfg 函数
放置位置：src/robot/config/env_cfgs.py
说明：这是**摘录**，依赖文件内其它部分（_make_base_env_cfg / get_robot_cfg /
      _simulation_cfg 等），不能单独 import 运行；集成时请把这段替换/粘贴回
      env_cfgs.py 对应位置。

本场地关键设计（改参数前务必读）：
  1) 地形 BoxSteppingStonesTerrainCfg：两条交错桩列，桩 200x200x200 mm，
     横向中心距 0.40 m（≈轮距 0.385），纵向中心距 0.60 m（>轴距 0.387，必须抬腿跨），
     两列错开 0.30 m，10 级难度沿桩高/中心距/错位/净空逐级放宽。
  2) 落点判据（2026-11-28 后新增，本场地的核心）：
     - landing_precision：轮心到桩心的归一化距离，窗口内线性给分
     - landing_edge：超过 edge_frac 开始线性罚
     两者都带"轮子真的接触到桩顶"的门控，避免惩罚正常跨缝动作。
  3) 支撑轮数 wheel_support / 后轮踩空 rear_gap_clearance 是本场地的关键约束。
"""
def stepstone_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """梅花桩通行任务：滚动 + 抬腿跨越，按 16DOF_RL.md 的课程/奖励/终止配置。

    - 地形：两条交错桩列（0.20 x 0.20 m 方桩），桩高/中心距/错位/噪声/缺桩
      随难度从“低矮密集”升到实场的 0.20 m / 0.60 m / 0.30 m 错位。
    - 观测：53 维本体状态 + 273 维局部高度扫描（21 x 13，1.6 x 1.0 m），
      对应讨论里的“局部高度图”，暂不接真实 LiDAR 噪声。
    - 成功：机身沿 +x 走过 STEPSTONE_GOAL_DISTANCE 进入终点平台。
    - 课程：通过整段地形升一级，几乎原地不动降一级；同时自适应抬高速度指令。
    """
    cfg = _make_base_env_cfg()
    cfg.scene.entities = {"robot": get_robot_cfg()}

    # ------------------------------------------------------------------ 地形
    cfg.scene.terrain = TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            size=STEPSTONE_TERRAIN_SIZE,
            border_width=2.0,
            num_rows=STEPSTONE_NUM_LEVELS,
            num_cols=1,
            curriculum=True,
            difficulty_range=(0.0, 1.0),
            sub_terrains={
                "stepstones": BoxSteppingStonesTerrainCfg(
                    proportion=1.0,
                    size=STEPSTONE_TERRAIN_SIZE,
                    start_length=STEPSTONE_START_LENGTH,
                    end_length=STEPSTONE_END_LENGTH,
                ),
            },
        ),
        # Round 5b: 0→7。环境等级由 `terrain_entity.py:341-348` 的
        # `torch.randint(0, max_init_level+1)` 初始化，写 0 就意味着**每轮续训的
        # 2048 个 env 全部从 0 级出生**，要花约 450 迭代才爬回 6 级 —— R1~R4 每轮
        # 约 40% 的 GPU 预算都花在爬坡上，训练在 7-9 级真正停留的时间几乎为零，
        # 这才是"地形均值连卡 4 轮 6.0~6.1"的根本原因，而不是奖励权重。
        # 改成 7 后每轮开局就均匀分布在 0~7 级（均值约 3.5），8/9 级由课程升级在
        # 几十个迭代内自然填满，续训能立刻在能力前沿攒梯度。回放分支仍用 max_level(9)。
        max_init_terrain_level=7,
    )
    cfg.scene.terrain.num_envs = 2048
    cfg.scene.terrain.env_spacing = 2.5

    # ------------------------------------------------------------- 观测补充
    # 讨论里的“局部高度图”：把 critic 的高度扫描同时喂给 actor。
    # 网格 1.6 x 1.0 m / 0.08 m -> 21 x 13 = 273 维；actor 观测 53 -> 326。
    cfg.observations["actor"].terms["height_scan"] = ObservationTermCfg(
        func=safe_height_scan,
        params={"sensor_name": "height_scanner"},
        clip=(-1.0, 1.0),
    )
    # 四轮接触标志（4 维）：让策略"看得见"哪个轮子踩实、哪个悬空。
    # 之前 actor 只有本体状态 + 高度图，看不到轮子接触，才会形成"后轮悬空、
    # 靠轮子撞桩边被带上去"的捷径（2026-10-28 用户回放反馈）。实机侧可用
    # 轮电机电流/扭矩估计接触。
    cfg.observations["actor"].terms["wheel_contact"] = ObservationTermCfg(
        func=safe_foot_contact,
        params={"sensor_name": "feet_ground_contact"},
    )

    # ------------------------------------------------------------------ 指令
    # 2026-10-28 p6 步态诊断：机身 y 精确锁死中线（std 0.023），轮子在 ±0.19 m、
    # 桩列在 ±0.20 m，差 1-3 cm 挂不上桩——策略从未学过"横移去压桩"。
    # 原因：rel_lateral_envs=0.10 意味着 90% 的时间指令是"纯直行"，横移从没被
    # 练过。放开到 0.35，并加宽 lin_vel_y，让"左右蹭桩"进入策略的探索空间。
    twist = cfg.commands["twist"]
    twist.heading_command = False
    twist.rel_heading_envs = 0.0
    twist.ranges.heading = None
    twist.rel_standing_envs = 0.05
    twist.rel_forward_envs = 0.60
    # Round 8: 0.35→0.45，lin_vel_y ±0.30→±0.60。
    # R7 把 stone_touch 6.5→8.5、all_stones 20→30 想靠奖励拉覆盖率，结果
    # coverage 只从 0.524 涨到 0.534，反而 base_too_low 从 0.167 退到 0.292
    # ——说明覆盖率**不是奖励给少了，是策略本身没有横移能力**：
    # 梅花桩 zigzag 每跳要横向挪 0.30 m（错位）而纵向只有 0.46 m，折算需要
    # 约 0.6 m/s 的横移速度，可指令上限才 0.3 m/s、且只有 35% 的环境拿到
    # 横移指令 → 另外 65% 的回合里 track_lin_vel_y 反而在奖励"y 速度=0"。
    # 这轮直接把横移指令的**供给**加宽（比例 45%、幅度到 ±0.6），让策略在
    # 赛道需要的那档横向速度上真正练过。
    twist.rel_lateral_envs = 0.45
    twist.rel_yaw_envs = 0.15
    twist.ranges.lin_vel_x = (0.3, 0.8)
    twist.ranges.lin_vel_y = (-0.6, 0.6)
    twist.ranges.ang_vel_z = (-0.4, 0.4)
    twist.resampling_time_range = (6.0, 6.0)

    # ------------------------------------------------------------ 重置与扰动
    cfg.events["reset_base"] = EventTermCfg(
        func=envs_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (0.0, 0.0),  # default_root_state 已是站姿高度，叠加会多掉 0.36 m
                "yaw": (-0.15, 0.15),
            },
            "velocity_range": {"x": (-0.1, 0.1), "y": (-0.05, 0.05), "z": (0.0, 0.0)},
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.events["push_robot"] = EventTermCfg(
        func=envs_mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(6.0, 10.0),
        params={"velocity_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, "asset_cfg": SceneEntityCfg("robot")},
    )
    # 轮子-地形摩擦随机化：削弱"轮子撞桩边被带上去"的捷径（实机上 15-20 cm
    # 硬桩边会把轮子弹开/卡住），迫使策略学会主动把轮子放到桩顶面。
    cfg.events["wheel_friction"] = EventTermCfg(
        func=envs_dr.geom_friction,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", geom_names=(".*_wheel_geom",)),
            "ranges": (0.8, 1.2),
            "operation": "scale",
        },
    )

    # ------------------------------------------------------------------ 奖励
    cfg.rewards = {
        # 主信号：目标方向的实际进度（用机身位移，不用轮速）
        # Round 9fix-p4（2026-11-28）：1.5→2.0。p3 把 min_support 放宽到 2 之后
        # 四项硬指标（后轮不落桩下 2.27%、支撑轮数 2.02、腿碰 0.113%、机身 0.3066）
        # 全部刷新历史最好，但前进从 4.49 m 掉到 3.57 m——抬腿多了、步频慢了。
        # 这一轮把前进的主信号加强，试着在保住抬腿成果的前提下把速度找回来。
        "forward_progress": RewardTermCfg(
            func=forward_progress,
            weight=2.0,
            params={"command_name": "twist", "target_speed": 0.8},
        ),
        "track_lin_vel_x": RewardTermCfg(
            func=track_linear_velocity_x,
            weight=0.8,
            params={"std": 0.25, "command_name": "twist"},
        ),
        # p10 实验（2026-10-28）：补横向速度跟踪。
        # 步态诊断显示机身 y 一直锁死赛道中线（std 0.02 m），轮子靠擦边挂桩
        # 才拿到 0.5 左右的覆盖率——奖励里此前只有 vx/yaw 跟踪，横移从未被
        # 激励过（rel_lateral_envs 虽然放开到 0.35，但策略不跟 y 指令）。
        # 加 0.5 权重的 vy 跟踪，让策略先在平地学会横移，再带到高级地形
        # 对准桩、稳落点，突破 6 级瓶颈。
        # Round 4: 0.5→0.9。Round 2/3 两轮都复现同一个模式：地形爬到 6.5~6.9 级后
        # 回落到 5.8~6.0 并卡死。6→7 级唯一的变化是错位 0.240→0.264、桩高 +1.2cm
        # （跨距都被 max_step=0.46 截断，同样是 460mm）。所以 7 级瓶颈是**横向对准**
        # 不行，不是跨远不行。原注释已写明 vy 跟踪就是为了破 6 级瓶颈，现在把它加重。
        "track_lin_vel_y": RewardTermCfg(
            func=track_linear_velocity_y,
            # Round 8: 0.9→1.2。横移指令幅度放宽到 ±0.6 之后，需要同步加大跟踪
            # 权重，否则新增的 ±0.3~0.6 档指令基本处在"跟不跟都无所谓"的状态。
            weight=1.2,
            params={"std": 0.25, "command_name": "twist"},
        ),
        "track_ang_vel_z": RewardTermCfg(
            func=track_angular_velocity_z,
            weight=0.4,
            params={"std": 0.4, "command_name": "twist"},
        ),
        # 姿态与高度：不要求完全水平，只惩罚异常俯仰/横滚和趴地
        "flat_orientation": RewardTermCfg(func=flat_orientation_l2, weight=-0.5),
        "body_height": RewardTermCfg(
            func=body_height_band,
            weight=-1.0,
            params={"min_height": 0.18, "max_height": 0.75},
        ),
        # 轮子空转 / 滑移：只在接触时统计
        "wheel_slip": RewardTermCfg(
            func=wheel_slip,
            weight=-0.05,
            params={"sensor_name": "feet_ground_contact", "wheel_radius": naming.WHEEL_RADIUS},
        ),
        # 能耗与平滑
        "joint_torques": RewardTermCfg(func=envs_mdp.joint_torques_l2, weight=-2.5e-5),
        "joint_power": RewardTermCfg(func=joint_power, weight=-2.0e-5),
        "action_rate": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.01),
        "joint_pos_limits": RewardTermCfg(func=envs_mdp.joint_pos_limits, weight=-5.0),
        # 机身撞桩（腿/轮与桩顶的接触不罚）
        "undesired_contacts": RewardTermCfg(
            func=undesired_contacts,
            weight=-0.5,
            params={"sensor_name": "body_collision", "threshold": 1.0},
        ),
        # 到终点给 +120，并按桩覆盖率打折。floor 从 0.2 收到 0：p4 轮（2048 环境
        # 1000 迭代，zigzag 地形）实测 coverage 精确卡在 0.50——策略走"同侧序列"
        # 每两根踩一根，跳过另一侧的桩。floor=0.2 时踩半 72 分 vs 踩满 120 分，
        # 差距只有 48，不够抵消掉坑风险，策略没有动力改步态。归零后踩半 60、
        # 踩满 120，配合 stone_touch 3.0，"每根都踩"的期望显著占优。
        # 权重从 60 提到 120：2026-10-27 那次训练里策略发现"在桩上慢慢蹭、耗到超时"
        # 比冲过终点更赚（每根桩 +5，16 根就 80），于是停在最后一根桩前不动。
        "goal_reached": RewardTermCfg(
            func=goal_with_coverage,
            weight=120.0,
            params={"distance": STEPSTONE_GOAL_DISTANCE, "floor": 0.0},
        ),
        # 摔倒比超时更糟：摔倒丢掉所有后续机会，超时至少还在桩上。
        # 反过来（超时 -20 > 摔倒 -10）会让“过不去就赶紧摔”成为最优解——2026-10-27
        # 20:02 那次训练里 stone_coverage 从 0.45 崩到 0.07、distance 卡在 1.5 m 就是这个。
        # 摔倒 -10（原来 -20）：过不去时“摔一下”不该比“站着耗到超时”贵太多，
        # 否则策略会在难等级上直接摆烂。超时罚见下面的 time_out（按距离缩放）。
        "is_terminated": RewardTermCfg(func=envs_mdp.is_terminated, weight=-10.0),
        # 超时罚按“离终点还差多少”缩放（越近越轻）：固定值会让“站在最后一根桩前发呆”
        # 和“站在起点发呆”一个价，而摔倒固定 -10 -> 摆烂更划算（2026-10-27 play 里就是这个）。
        "time_out": RewardTermCfg(
            func=time_out_penalty,
            weight=1.0,
            params={"distance": STEPSTONE_GOAL_DISTANCE, "scale": -10.0},
        ),
        # 反“飞越”：桩区上空且四轮全部离地 -> -1/步；超过指令速度 0.3 m/s 以上按超出量罚。
        "airborne": RewardTermCfg(
            func=airborne_over_pit,
            weight=-1.0,
            params={
                "x_min": STEPSTONE_START_LENGTH,
                "x_max": STEPSTONE_TERRAIN_SIZE[0] - STEPSTONE_END_LENGTH,
            },
        ),
        "overspeed": RewardTermCfg(
            func=overspeed, weight=-0.5, params={"command_name": "twist", "margin": 0.3}
        ),
        # 赛事规则：每个桩顶面必须至少被足底（轮子）接触一次。
        # 单桩权重 3 -> 5：p6 步态诊断显示机身锁死中线、轮子离桩 1-3 cm 挂不上，
        # 需要更强的单桩收益推动策略学会"横移压桩"。42 根全踩 210 vs 半踩 105。
        # Round 4: 5.0→6.5。R3 踩边率已从 33% 降到 12%，落点质量改善后要开始
        # 奖励"往前走、每根都踩"——stone_touch 是唯一按"踩到新桩"给分的项，
        # 提高它能抵掉高级地形下"不敢迈步"的保守倾向。
        # Round 7: 6.5→8.5。R5b/R6 两轮把 stone_coverage 稳稳钉在 0.52，等于"每两根
        # 只踩一根、走同侧直线"。赛事规则是**每个桩顶面至少被足底接触一次**，覆盖率
        # 直接决定成绩；而且 7-9 级相对 6 级唯一的差别是错位 0.264→0.300（跨距全都是
        # 被 max_step 截断的 460mm），所以"不会侧向踩到另一排桩"同时也是卡在 6 级的
        # 症结。提高单桩收益，让"左一根右一根"的 zigzag 明显比直着走更划算。
        # Round 7: 6.5→8.5（拉覆盖率）；Round 8: 8.5→7.0（部分回撤）。
        # R7 实测覆盖率只 +0.01、而 base_too_low 从 0.167 退到 0.292，等于用摔倒
        # 换来的覆盖几乎为零；本轮把主攻方向换成"横移指令供给"（见上面 twist），
        # 覆盖奖励回撤一半作为辅助，避免同时两个方向用力把稳定性吃掉。
        "stone_touch": RewardTermCfg(func=stone_touch_reward, weight=7.0),
        # Round 7: 20→30；Round 8: 30→25。
        "all_stones": RewardTermCfg(func=all_stones_bonus, weight=25.0),
        # 四轮支撑：桩区内每步按"有几个轮子真正站在桩顶面上"给分（0-4），
        # 桩区外返回 0。判定是"轮心在桩顶矩形内 + 轮底贴桩顶"，蹭桩边不算。
        # 2026-10-28 用户两次回放反馈：先有后轮悬空、靠轮子撞桩边被带上去；
        # 收紧后又有轮子落在桩边缘/桩间缝隙（视觉像"陷进桩里"）。严格判定
        # 逼策略把轮子放到桩顶面中心。
        "wheel_support": RewardTermCfg(
            func=wheel_support_reward,
            # Round 2: 0.2→0.8，加强四轮支撑奖励，鼓励轮子始终落在桩顶面上
            # Round 6: 0.8→1.2。R5b 关掉等级封顶+开局 0~7 级分布后，整轮都真正停在
            # 7-9 级前沿（max 恒定 9、base_too_low 0.500→0.348 明显改善），但代价是
            # 单步步态质量指标回落：支撑率归一化 0.263→0.206、落点精度 0.193→0.147。
            # 这是"换到更难地形上练"的正常现象，现在把 R2/R3 已经验证有效的两个
            # 形态项继续往上推，让策略在 7-9 级上重新把步态做标准。
            # Round 9fix（2026-11-28）：1.2→0.4，对齐 R8 训练时的真实快照
            # （params/env.yaml 里就是 0.4）。当前代码里的 1.2 是 R10 之后才改的，
            # 不在任何一轮的记录里，续训前必须回退。
            weight=0.4,
            # 平均踩边率卡在 24-27%。支撑判定已收紧到中心带（support_frac 0.75），
            # 加大支撑权重让"踩准"比"跨远"更值钱，推动四条腿同时收敛。
            params={
                "x_min": STEPSTONE_START_LENGTH,
                "x_max": STEPSTONE_GOAL_DISTANCE,
                # 0.5：轮心必须落在桩心 ±5 cm 内才算支撑（2026-10-30 收紧：p27 落点
                # 中位数 4-6 cm、边缘率仍 20%+，让"踩边"连支撑分都拿不到，与
                # landing_precision、landing_edge 三管齐下逼四条腿都落桩心）。
                "support_frac": 0.5,
            },
        ),
        # 持续悬空惩罚：桩区内支撑轮数 < 3 连续超过 5 步（0.1 s）才罚，
        # 跳桩瞬间的正常腾空不罚。
        # Round 2: -0.5 → -3.0，强制轮子抬起不落到桩下
        "wheel_airtime": RewardTermCfg(
            func=wheel_airtime_penalty,
            # Round 10: -3.0→-4.0。这是用户第一优先级的硬指标（"轮子走的时候一定要
            # 抬起来，不得落到比梅花桩还低的地方"）。R6/R7 的事件率是 0.0116/0.0113，
            # R8 因为放宽横移指令退化到 0.0148——横移变快了，轮子在空中待得久了。
            # R9 已证明课程加压换不来难度，稳定性余额（R8 摔倒率 4%）不如花在
            # 这条硬指标上：把悬空罚加严，逼策略在横移时也保持至少 3 轮贴地。
            # Round 9fix（2026-11-28）：-4.0→-3.0。R10 把悬空罚加严到 -4.0，但独立
            # 体检（同一随机种子、30 秒满回合）显示它并没能换来"后轮不踩空"——
            # 后轮踩空率反而从 2.60% 恶化到 4.44%、腿碰地形从 0.23% 涨到 1.81%。
            # 所以回到 R8 的真实值 -3.0 重新出发。
            weight=-3.0,
            params={
                "sensor_name": "feet_ground_contact",
                "x_min": STEPSTONE_START_LENGTH,
                "x_max": STEPSTONE_GOAL_DISTANCE,
                # Round 9fix-p3（2026-11-28）：3→2。前两轮证明"收紧落点判据"治不了
                # "后轮低于桩顶"（严口径 2.60%→3.99%→3.23%，始终没超过 R8）。根因
                # 不在落点，而在抬腿高度：min_support=3 要求**至少 3 个轮子同时贴地**，
                # 等于反过来限制了轮子能抬多高。用户原话是"轮子走的时候一定要抬起来，
                # 不得落到比梅花桩还低的地方"——放宽到 2，方向上才是顺着这个诉求。
                "min_support": 2,
                "grace_steps": 5,
            },
        ),
        "pit_contact": RewardTermCfg(
            func=pit_contact_penalty, weight=-2.0, params={"min_wheels": 2}
        ),
        # 后轮过缝惩罚：后两轮位于桩间隙正上方、轮心低于最近桩顶 7 cm 以内时，
        # 按每轮缺口深度罚。跨桩时后腿抬不够、轮子蹭到桩侧或掉进缝里，实机上会卡死；
        # 前轮抬高没问题（探针统计前轮全程高于桩顶 7 cm），所以只罚后两轮。
        "rear_gap_clearance": RewardTermCfg(
            func=rear_gap_clearance_penalty,
            # Round 9fix：-30.0→-50.0，对齐 R8 真实快照（env.yaml 里是 -50.0）。
            weight=-50.0,
            params={
                "margin": 0.07,
                "x_min": STEPSTONE_START_LENGTH,
                "x_max": STEPSTONE_GOAL_DISTANCE,
            },
        ),
        # 落点精度：踩在桩顶面上时，轮心离桩心越近分越高（边缘≈0，中心=1），
        # 三次方曲线让"搭边"几乎得不到分，逼策略把轮子放到桩顶中央带。
        # 要求机身速度 > 0.25 m/s 才计分——否则策略会站在桩上不动刷分
        # （p15 实测：ep_len 1405 步、goal 归零、landing_precision 0.76/步）。
        # 权重 0.5→0.8：p17 的 FR 仍有 43% 踩边，需要更强的"往中心落"引力。
        # Round 3: 0.8→1.2。Round 2 归一化后 landing_edge 违反率 33%（-0.1001/-0.3），
        # 仍是最大的单步缺陷；地形爬到 6.93 级后在 7 级回落到 5.8~6.0，瓶颈就是踩边。
        # 与 landing_edge 同向加强（中心=给分、边缘=罚分），形成一致梯度而非互相抵消。
        "landing_precision": RewardTermCfg(
            func=landing_precision_reward,
            # Round 6: 1.2→1.6。与 wheel_support 同步上调：R5b 把训练真正推到 7-9 级
            # 前沿后，单步落点精度从 0.193 回落到 0.147（踩边仍是高级地形掉下去的主因）。
            # 中心给分与 landing_edge 边缘罚分保持同向上调，梯度方向一致不互相抵消。
            # Round 9fix-p2（2026-11-28）：2.2→1.6 回退。p1 把给分提到 2.2 后，策略为了挤进
            # 桩心去撞桩，腿碰地形率从 0.15% 恶化到 0.80%（5.4 倍），而"后轮不低于
            # 桩顶"没有改善。说明"加奖金"这条路走不通，回退。
            # Round 9fix-p8（2026-11-28）：1.6→2.0，同时把给分窗口 0.5→0.75 放开。
            # 用户回放指出"后轮已稳在中心、前轮还压在半边"，而前轮偏离 5~7 cm 正好
            # 在旧窗口外、梯度恒为 0——这才是权重调不动的真因。窗口放开后权重才有意义。
            # Round 9fix-p9（2026-11-28）：窗口 0.75→0.3、权重 2.0→2.5。p8 实测证明
            # 宽窗口只喂饱了后轮（4.44→3.93 cm），前轮纹丝不动（4.68→4.62 cm）——
            # 因为 7.5 cm 的窗口让前轮"停在 4.7 cm 也有分拿"，等于给了它一个及格点。
            # 用户原话是"落脚点改为桩中心 ±3 cm，超了就狠狠地罚"，所以窗口收到 3 cm：
            # 3 cm 内才给分，3 cm 外纯吃罚，平衡点从 5.7 cm 挪到 3 cm。
            # Round 9fix-p10（2026-11-28）：p9（窗口 0.3 + 罚 -2.0）实测全面倒退——
            # 前 4.56 / 后 4.47 cm、±3cm 内只有 33.9%、两轮上桩率掉 7~8 个百分点。
            # 罚重了它就直接减少上桩（又一次回避）。所以回到 p8 的温和力度，改为
            # **前/后轮分开设窗口**（用户同意）：前轮探路落地晚、给它宽一点的梯度
            # （0.6 ≈ 6 cm），后轮承重、收紧到 0.45（≈4.5 cm）。
            weight=2.2,
            params={
                "x_min": STEPSTONE_START_LENGTH,
                "x_max": STEPSTONE_GOAL_DISTANCE,
                "bottom_tol": 0.06,
                "min_speed": 0.25,
                "precision_window": 0.6,
                "front_window": 0.6,
                "rear_window": 0.45,
            },
        ),
        # Round 9fix：合格落点从"桩心 ±5 cm"收紧到"桩心 ±3 cm"（edge_frac 0.5→0.3），
        # 并把 min_speed 门控关掉（0.25→0.0）——门控会让"站着不动"同时免疫本项
        # 惩罚，罚一加重，躺平就成了最优解。判据本身同步改成"轮心出桩面也照罚"。
        # 权重只有在 task_16dof.py 的 LANDING_EDGE_WEIGHT 里改才生效（会被它覆盖）；
        # 实测 -3.0 会触发"摔倒逃避"，最终采用 -0.8，原因见那里的注释。
        "landing_edge": RewardTermCfg(
            func=landing_edge_penalty,
            weight=-0.8,
            params={
                "x_min": STEPSTONE_START_LENGTH,
                "x_max": STEPSTONE_GOAL_DISTANCE,
                "bottom_tol": 0.06,
                "edge_frac": 0.3,
                "min_speed": 0.0,
            },
        ),
    }

    # ---------------------------------------------------------------- 终止
    cfg.terminations = {
        "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
        "bad_orientation": TerminationTermCfg(
            func=envs_mdp.bad_orientation, params={"limit_angle": math.radians(60.0)}
        ),
        "base_ground_contact": TerminationTermCfg(
            func=velocity_mdp.illegal_contact, params={"sensor_name": "base_ground_contact"}
        ),
        "base_too_low": TerminationTermCfg(func=base_too_low, params={"min_height": 0.12}),
        # 到达终点平台：成功终止（不是 time-out，值函数不做 bootstrap）
        "goal_reached": TerminationTermCfg(
            func=reached_goal, params={"distance": STEPSTONE_GOAL_DISTANCE}
        ),
        "nan_detection": TerminationTermCfg(func=envs_mdp.nan_detection),
    }

    # ---------------------------------------------------------------- 课程
    cfg.curriculum = {
        "terrain_levels": CurriculumTermCfg(
            func=stepstone_levels,
            # Round 5: 奖励调了 3 轮（R2/R3/R4），地形均值始终卡在 6.0~6.1，
            # 峰值 6.5~6.9 后必回落。查课程源码 `mdp/stepstone.py:192-201`：
            # 升级要求走完整段（dx > goal），降级只要 dx < goal*min_progress_frac
            # 连续 demote_after 次。也就是在 7 级骑虎难下时，2 次不达标就降回去，
            # Round 9: 课程加压。R8 把 base_too_low 压到 0.0417（只剩 4% 回合摔倒）、
            # bad_orientation 0.0，可地形均值仍只有 5.91、distance 6.75——
            # 说明"不摔"已经做到了，缺的是**难度供给**：策略待在 5.9 级的舒服区，
            # 每次爬上去又被 demote 回来，从没被真正压在 7-9 级攒够梯度。
            # 现在稳定性有余额了，把这笔余额花在课程上：更容易升级、更难被降级。
            # Round 9 实测 = 失败实验：地形均值 5.912→5.927（没动）、distance 6.75→6.35、
            # base_too_low 0.042→0.261、goal 3.625→3.087、coverage 0.550→0.541。
            # 说明 5.9 这个均值**不是课程压出来的**，而是一个结构性平衡点（爬上去就会
            # 在 7-9 级失败/超时被打回来，换多少次课程参数都是同一个均衡）。
            # → Round 10 全部回退成 R8 的值。
            params={
                "goal_distance": STEPSTONE_GOAL_DISTANCE,
                "min_progress_frac": 0.85,
                "demote_after": 3,
                "promote_prob": 0.6,
                # Round 5b: steps_per_level 1200→0（关掉等级封顶）。
                # 封顶用 `env.common_step_counter` 算 level_cap，而**续训时该计数器从 0
                # 重新开始**，导致每一轮新训练开头把全部环境强制撒回 0 级，要花约
                # 450 轮才爬回 6 级（R2/R3/R4/R5 全部如此，约 40% GPU 时间浪费在爬坡）。
                # 关掉后 level_cap = max_terrain_level（9），环境由地形生成器初始化成
                # 0~9 级随机分布，续训能真正继承上一轮难度分布、有机会在 7-9 级持续攒梯度。
                "steps_per_level": 0,
            },
        ),
        "command_x_levels": CurriculumTermCfg(
            func=command_levels_adaptive,
            params={
                "command_name": "twist",
                # 用**指令跟踪**项归一化（exp(-err²/std²) ∈ [0,1]）。用 forward_progress
                # 会把“跑得越快”当成“跟踪越好”，速度课程直接打满（实测 ema=3.03）。
                "reward_term_name": "track_lin_vel_x",
                "axis": "x",
                "initial_range": (0.3, 0.5),
                "delta_command": 0.05,
                "target_ratio": 0.6,
                "ema_alpha": 0.5,
            },
        ),
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
            "goal_reached": MetricsTermCfg(
                func=goal_reached_bonus, params={"distance": STEPSTONE_GOAL_DISTANCE}
            ),
            "stone_coverage": MetricsTermCfg(func=stone_coverage),
        }
    )

    cfg.episode_length_s = STEPSTONE_EPISODE_S
    cfg.decimation = 4
    cfg.seed = 42
    cfg.sim = _simulation_cfg(ccd_iterations=16)

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.curriculum = {}
        if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
            # 回放时不再更新等级，但**保留初始等级采样**：env 会均匀分布到 0..N 级，
            # 这样 120 个环境能同时看到从最简单到赛道规格的全部难度。
            # 用环境变量 STEPSTONE_PLAY_MAX_LEVEL 控制上限（默认 9 = 最高级）。
            max_level = int(os.environ.get("STEPSTONE_PLAY_MAX_LEVEL", "9"))
            cfg.scene.terrain.terrain_generator.curriculum = True
            cfg.scene.terrain.terrain_generator.num_rows = max(1, max_level + 1)
            # 每个 env 一个 patch（120 环境 = 10 行 x 12 列），否则同一等级的 env 会叠在同一个出生点。
            cfg.scene.terrain.terrain_generator.num_cols = int(os.environ.get('STEPSTONE_PLAY_COLS', '12'))
            cfg.scene.terrain.terrain_generator.border_width = 4.0
            cfg.scene.terrain.max_init_terrain_level = max_level
