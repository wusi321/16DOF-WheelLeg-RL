# Odin1 感知接口与 16 DOF 轮足 RL 集成（设计与调查）

本目录是「Odin1 → 局部地形观测 → RL 策略」这条链路的**接口调查、设计、仿真/真机映射与实施计划**。
本轮任务只做**分析与设计**：没有修改 `src/robot/` 下任何 RL 环境代码，也没有新增 Python 依赖；
所有「实现」都写成可直接落地的规格，等确认后再动代码。

## 五份交付物

| 文档 | 内容 | 状态 |
| --- | --- | --- |
| [01_odin1_interface.md](01_odin1_interface.md) | Odin1 ROS 2 接口调查：版本、ROS 2 发行版、topic、消息类型、坐标系、频率、QoS、启动参数、数据格式、文档与代码不一致点 | 已完成（逐条附来源） |
| [02_unified_perception_interface.md](02_unified_perception_interface.md) | 统一感知接口：数据契约、进程边界（训练进程不依赖 ROS 2）、源适配器、时序与失效处理 | 已完成 |
| [03_sim2real_mapping.md](03_sim2real_mapping.md) | MuJoCo 观测 ↔ 真实 Odin1 观测的映射，以及误差/噪声/稀疏/遮挡/延迟/运动畸变的建模方案 | 已完成 |
| [04_local_terrain_obs.md](04_local_terrain_obs.md) | 局部地形观测定义：坐标系、栅格、分辨率、范围、无效值、时间戳、桩顶/边缘/落点特征 | 已完成 |
| [05_roadmap.md](05_roadmap.md) | 三阶段实施计划、验收标准、风险与回滚点 | 已完成 |

## 一句话结论

- **Odin1 能用，但接口细节必须按驱动源码核对**：官方 README 的 topic 表里至少有 2 个名字与代码不一致（`image_undistort` / `odometry_high`），代码里还有 README 没列的 topic。
- **训练进程绝不能碰 ROS 2**：真实数据在**独立进程**里转成与仿真同构的 `TerrainFrame`（numpy/torch），训练只吃数组；录制数据用离线 npz 回放。
- **策略观测必须 base 系、与全局地图无关**：`map`/`odom` 只服务于导航与航点生成，不进入策略输入。
- **仿真侧的观测语义已经定死了**：`h = z_base − z_ground`，21×13 @ 0.08 m，`clip(-1, 1)`，未命中 → 1.0。真实侧必须复现这套语义，而不是另发明一套。

## 事实基线（用于交叉核对）

| 项 | 值 |
| --- | --- |
| 驱动仓库 | `manifoldsdk/odin_ros_driver`，Apache-2.0，HEAD `f51051f2d861f7643d4d33d2ade2952efe1a4672` |
| 驱动版本 | v0.14.4（`package_ros2.xml`） |
| 设备固件 | v0.14.0（README §1） |
| 官方 Wiki | <https://manifoldtechltd.github.io/wiki/odin_series/odin1/> |
| 本仓库当前感知实现 | 仅 MuJoCo 射线高度扫描 `height_scanner`（`src/robot/config/env_cfgs.py:203-212`） |

## 待验证清单（不允许当成已知规格使用）

| # | 待验证项 | 为什么重要 | 验证方法 |
| --- | --- | --- | --- |
| V1 | `cloud_slam` 的 `header.frame_id` 与点坐标实际所在坐标系（代码写 `map`，Wiki §5.4 说在 odom 系发布） | 直接决定是否要额外做 odom→map 变换 | 重定位模式录一段，比较 cloud_slam 与 TF 是否自洽 |
| V2 | `T_imu_lidar` 平移：代码 `x=+0.00347` vs Wiki `x=-0.02663` | 3 cm 的系统性偏差会直接进高度图 | 以运行期 `/tf` 的 `imu→lidar` 为准，实测比对 |
| V3 | `cloud_raw` 是否已在设备端做运动补偿（`offset_time` 如何用） | 决定要不要做去畸变，以及轮足 1–2 m/s 下的高度图拖影 | 匀速旋转设备录数据，看同一平面的残差 |
| V4 | 实际安装位姿（Odin1 在机身上的高度、俯仰、前后偏移） | 决定 120°×90° FOV 覆盖到地面的范围与最小可测距离 | 整机装配后实测标定 |
| V5 | `odometry` 的实际发布频率（Wiki 说 10 Hz）与 `odometry_highfreq` 的 400 Hz 抖动 | 决定插值/预测策略 | `ros2 topic hz` + bag 时间戳统计 |
| V6 | 重定位模式下行进中 `odom→map` 的更新频率与跳变幅度 | 若用于导航需处理跳变；策略不用 | 长距离录制统计 |
| V7 | `use_host_ros_time=2` 的 NTP 同步在机器人整机上是否稳定（30 帧滑窗） | 多传感器时间对齐 | 同时录 IMU 与轮速，比对相位 |
| V8 | `confidence` 的实际分布与最佳阈值（文档建议 30–35） | 决定点云过滤强度 | 录 bag 统计，按距离分层 |
