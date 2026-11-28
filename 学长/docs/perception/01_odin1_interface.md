# 01 · Odin1 ROS 2 接口调查

> 所有结论只来自两处可核对来源：
> **(A)** 驱动仓库 `manifoldsdk/odin_ros_driver`，HEAD `f51051f`（2026-09 最后推送），本文引用的
> 文件与行号都按这个 commit；
> **(B)** 官方 Wiki <https://manifoldtechltd.github.io/wiki/odin_series/odin1/>（中文版，页面名标注在引用处）。
>
> 凡两处冲突或只有一处提及的，都进第 9 节「不一致与待验证」，**不当作规格使用**。

## 1. 版本与支持矩阵

| 项 | 结论 | 来源 |
| --- | --- | --- |
| 驱动版本 | **v0.14.4** | `package_ros2.xml` 第 5 行 `<version>` |
| 配套设备固件 | **v0.14.0** | README §1 "Required device firmware version" |
| ROS 2 发行版 | **Foxy**（Ubuntu 20.04）、**Humble**（Ubuntu 22.04，官方推荐） | README 第 13 行 "ROS 2 (LTS Release: Humble recommended)"；§2.1 |
| 其他 | Ubuntu 18.04 不支持；24.04/Jazzy 未官方支持，README 提到可按 `tf_extra_publish_rate` 等建议手动适配 | README §2.1、§5.13 |
| 构建 | `colcon` + `ament_cmake`，依赖 `rclcpp / sensor_msgs / nav_msgs / pcl_conversions / cv_bridge / tf2_ros / image_transport` 等 | `package_ros2.xml` |
| 许可证 | Apache-2.0 | `LICENSE`、GitHub 元数据 |

运行（ROS 2）：

```bash
ros2 launch odin_ros_driver odin1_ros2.launch.py
```

launch 会起 5 个节点：`host_sdk_sample`（主数据）、`pcd2depth_ros2_node`（深度补全 demo）、
`cloud_reprojection_ros2_node`（点云重投影）、`image_overlay_node`（叠加显示）、`rviz2`。
来源：`launch_ROS2/odin1_ros2.launch.py`。

> 注意：这 5 个节点**默认全部启动**，其中 3 个是 demo 且很吃 CPU。真机集成时应自己写 launch
> 只起 `host_sdk_sample`，或把 `senddepth / sendreprojection / sendoverlay` 置 0。

## 2. Topic 全表（以驱动代码为准）

以下 topic 名在代码里写作 `"odin1/xxx"`（无前导 `/`，节点无 namespace 时解析为 `/odin1/xxx`）。
**这一节是本文最重要的一张表**：官方 README 的「4.3 ROS topics」表与代码有明显出入（见第 9 节）。

| Topic | 类型 | `header.frame_id` | 默认频率 | 发布者（代码位置） |
| --- | --- | --- | --- | --- |
| `/odin1/imu` | `sensor_msgs/Imu` | `imu` | **400 Hz** | `host_sdk_sample.h:1932` |
| `/odin1/odometry` | `nav_msgs/Odometry` | `odom`（child `imu`） | **10 Hz** | `host_sdk_sample.h:1936`、`:1286-1287` |
| `/odin1/odometry_highfreq` | `nav_msgs/Odometry` | — | **400 Hz** | `host_sdk_sample.h:1937` |
| `/odin1/cloud_raw` | `sensor_msgs/PointCloud2` | `lidar` | **10 Hz**（`dtof_fps: 100`），可设 14.5 Hz | `host_sdk_sample.h:1934`、`:581,:650` |
| `/odin1/cloud_slam` | `sensor_msgs/PointCloud2` | `map`（待验证，见 V1） | 10 Hz（跟随 dtof） | `host_sdk_sample.h:1935` |
| `/odin1/cloud_render` | `sensor_msgs/PointCloud2` | —（跟随 SLAM 云） | 10 Hz | `host_sdk_sample.h:1940` |
| `/odin1/image` | `sensor_msgs/Image`（bgr8） | — | 10.26 / 14.47 / 28.92 Hz（三档） | `host_sdk_sample.h:1933` |
| `/odin1/image/compressed` | `sensor_msgs/CompressedImage`（jpeg） | — | 同上 | `host_sdk_sample.h:1941` |
| `/odin1/image/undistorted` | `sensor_msgs/Image` | — | 同上（`sendrgbundistort: 0` 默认**关**） | `host_sdk_sample.h:1942` |
| `/odin1/image/intensity_gray` | `sensor_msgs/Image`（mono8） | `map` | 跟随 dtof（`pubintensitygray: 0` 默认关） | `host_sdk_sample.h:1943`、`:780` |
| `/odin1/path` | `visualization_msgs/MarkerArray` | — | 跟随 odometry | `host_sdk_sample.h:1938` |
| `/odin1/camera_pose_visual` | `visualization_msgs/MarkerArray` | — | 跟随 odometry | `host_sdk_sample.h:1939` |
| `/odin1/wiwc` | `nav_msgs/Odometry` | — | 跟随 odometry | `host_sdk_sample.h:1944` |
| `/odin1/depth_img_competetion` | `sensor_msgs/Image`（深度） | — | 10 Hz（`senddepth: 0` 默认**关**） | `depth_image_ros2_node.cpp:27` |
| `/odin1/depth_img_competetion_cloud` | `sensor_msgs/PointCloud2` | — | 10 Hz（同上） | `depth_image_ros2_node.cpp:28` |
| `/odin1/reprojected_image` | `sensor_msgs/Image` | — | 跟随 SLAM 云（`sendreprojection: 0` 默认关） | `cloud_reprojection_ros.cpp:89` |
| `/odin1/overlay_image` | `sensor_msgs/Image` | — | 同上（`sendoverlay: 0` 默认关） | `image_overlay_node.cpp:46` |

服务（`srv/`）：`/odin1/get_ae`、`/odin1/get_awb`、`/odin1/set_ae`、`/odin1/set_awb`、
`/odin1/get_device_logs`、`/odin1/save_map`、`/odin1/reset_algo`（`host_sdk_sample.cpp:2274-2275`）。
**RL 只用流数据，不需要任何服务**。

## 3. 消息与数据格式

### 3.1 `cloud_raw`（点云，最关键的输入）

字段顺序与类型（README §4.4，与 `include/lidar_api_type.h` 中的自定义点类型一致）：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `x, y, z` | `float32` | 米，传感器（`lidar`）坐标系 |
| `intensity` | `uint8` | 反射强度 0–255 |
| `confidence` | `uint16` | 置信度，典型场景 0–1300，**建议过滤阈值 30–35** |
| `offset_time` | `float32` | 相对该帧基准时间戳的偏移，单位 **秒** |

- README §4.4 明确给出了 PCL 注册代码（`ls_ros::Point`），说明这是非标准 PointCloud2 布局；
  用 `sensor_msgs_py.point_cloud2.read_points` 时必须显式列出 fields。
- `cloud_slam` / `cloud_render` 是另一套：`x, y, z, rgb`（`rgb` 为 `float32` 打包），
  已经被 SLAM/渲染处理过，**不是原始测距**。

### 3.2 深度模组（`cloud_raw` 的物理来源）

来源：Wiki「14. 技术参数」+ 仓库 `config/control_command.yaml`。

| 参数 | 值 |
| --- | --- |
| 深度分辨率 | **240 × 180** |
| 测距范围 | 0.2 m – 30 m @10% 反射率 / 70 m @90% 反射率（<500 lux） |
| 测距精度 | **±3 cm @1σ**（5 m、80% 朗伯板、连续 20 帧）；5–30 m 1σ < 1% |
| 点频 | 最高 **700 000 pts/s** |
| FOV / 角分辨率 | **120° H × 90° V** / **0.5° × 0.5°** |
| 帧率 | 默认 **10 FPS**，最高 15 FPS（`dtof_fps: 100` 或 `145`） |
| 波长/安全 | 940 nm，Class 1 |

### 3.3 RGB（本项目暂不用）

1600×1296、全局曝光、129°H×104°V、10/15 fps、`rgb_format: 1`（MJPEG）。

## 4. 坐标系与 TF 树（影响变换，必须按运行期 TF 为准）

Wiki「5. 数据输出」§5.1–5.4 定义三个硬件坐标系：**I**（IMU）、**L**（点云）、**C**（相机，OpenCV 光学约定）。

代码实际广播的 TF（`host_sdk_sample.h`）：

| 父子 | 何时发布 | 代码位置 |
| --- | --- | --- |
| `odom → imu` | 每帧 odometry（`send_odom_baselink_tf: 1` 时） | `:1576-1577` |
| `imu → lidar` | 同上（外参固定） | `:1596-1597` |
| `odom → camera_0` | 同上（由 odometry 位姿 + 外参算出） | `:1628-1629` |
| `odom → map` | **仅重定位模式**（`custom_map_mode: 2`）且重定位成功后 | `:1556-1557` |

Wiki §5.4 的三条要点（与代码互相印证）：

1. **IMU 系可直接当作 `base_link`**（接入 ROS 时把它当基座，或自己发一条 `base_link → imu` 静态变换）。
2. TF 默认按**里程计频率 10 Hz** 发布；`tf_extra_publish_rate > 0` 时额外补发。
3. `odom → map` 的**父子方向与 REP-105 的 `map → odom` 相反**，接导航栈要自己取逆再广播。

外参 `T_imu_lidar`：Wiki §5.2 给出平移 `(-0.02663, 0.03447, 0.02174)`、旋转为单位阵；
**驱动代码里的常量是 `(0.00347, 0.03447, 0.02174)`**（`host_sdk_sample.h:1890` 附近），两者差 3 cm，
属于必须验证的矛盾项（V2）。任何情况下都应读运行期 TF，不硬编码。

## 5. 频率、时间戳与同步

| 项 | 结论 | 来源 |
| --- | --- | --- |
| IMU | 400 Hz；`enable_imu_smooth: 1` 时由 SDK 以固定 400 Hz 发送（默认 0，按设备节奏发） | `control_command.yaml`（`imu_smooth_frequency: 400`）、README §5.13 |
| 高频里程计 | 400 Hz（`/odin1/odometry_highfreq`） | README §5.13 |
| 里程计 | 10 Hz | Wiki §5.4 |
| 点云/图像 | 10 Hz（可 14.5 Hz） | `dtof_fps / rgb_fps` |
| 时间戳来源 | `use_host_ros_time`：0=设备开机时间（默认、推荐）；1=主机收到时间；2=NTP-like 软同步，30 帧滑窗补偿后统一到主机时间轴 | `control_command.yaml`、Wiki「12. 时间同步」 |

对感知链路的影响：选 **0**（设备时间）或 **2**（与主机对齐）。如果机器人本体也用自己的时钟，
建议 2，让点云、IMU、轮速都在主机时间轴上；只用 1 会让时间戳带上 USB/DDS 抖动。

## 6. QoS（订阅时最容易踩坑的地方）

驱动发布端（`host_sdk_sample.h:1923-1930`）：

| 类 | QoS |
| --- | --- |
| `qos_small`（imu、odometry_highfreq） | `RELIABLE`、`VOLATILE`、**depth 4000** |
| `qos_sensor`（点云、图像、odometry、path…） | `RELIABLE`、`VOLATILE`、**depth 5** |

README §5.13 记录过一个真实问题：`ros2 bag record` 默认 `keep_last / depth 10`，在 400 Hz 下只缓冲
约 25 ms，录制端一阻塞就**在订阅端**静默丢帧（DDS 丢最旧样本，发布端无感）。录制 bag 必须：

```bash
ros2 bag record -o odin1_bag \
  --qos-profile-overrides-path qos_4000.yaml   # 至少 imu / odometry_highfreq 用 depth: 4000
```

## 7. 启动参数（`config/control_command.yaml`）

与本项目相关的键（其余见原文件）：

| 键 | 默认 | 作用 |
| --- | --- | --- |
| `use_host_ros_time` | `0` | 时间戳策略（0/1/2） |
| `sendimu` / `sendodom` / `senddtof` / `sendrgb` | `1` | 各流开关 |
| `sendcloudslam` / `sendcloudrender` | `1` | SLAM 云/渲染云开关 |
| `senddepth` / `sendreprojection` / `sendoverlay` | `0` | demo 开关，默认关 |
| `dtof_fps` | `100` | 100=10 Hz，145=14.5 Hz |
| `rgb_fps` | `100` | 与 `dtof_fps` 联动，三档 |
| `cloud_raw_confidence_threshold` | `35` | 点云置信度过滤阈值 |
| `custom_map_mode` | 见文件 | 0=里程计、1=建图（回环）、2=重定位 |
| `send_odom_baselink_tf` | `1` | `odom→imu` 等 TF 开关 |
| `tf_extra_publish_rate` | `0` | TF 补发频率（Jazzy 建议 50–100） |
| `strict_usb3.0_check` | `0` | 是否强制 USB3 |

## 8. 对本项目的直接结论

1. **感知主输入选 `/odin1/cloud_raw`**，不选 `cloud_slam`：
   后者是 SLAM 优化后的地图系点云（含回环/重定位跳变），不适合做“脚下 1 m 内地形”的实时输入；
   `cloud_raw` 是纯测距、带 `confidence` 和 `offset_time`，正好对应仿真里的“一次射线扫描”。
2. **不要用 `/odin1/image` 做深度**：RGB 无深度，深度补全 (`depth_img_competetion`) 是 demo、
   默认关闭且依赖 CPU 重投影，延迟不可控。
3. **姿态从 `/odin1/odometry_highfreq`（400 Hz）+ `/odin1/imu`** 取，点云配时间同步用
   `offset_time` 去畸变；`/odin1/odometry`（10 Hz）只作低频校验。
4. **策略不接触 `map` 系**：`map` 只在导航/航点层使用（见 04 文档）。
5. 训练进程**完全不依赖 ROS 2**：见 02 文档的进程边界设计。

## 9. 文档与代码不一致 / 待验证

| # | 现象 | 证据 | 处理 |
| --- | --- | --- | --- |
| D1 | README 表写 `odin1/image_undistort`，代码是 **`odin1/image/undistorted`** | README §4.3 vs `host_sdk_sample.h:1942` | 以代码为准；订阅前 `ros2 topic list` 复核 |
| D2 | README 表写 `odin1/odometry_high`，代码是 **`odin1/odometry_highfreq`** | README §4.3 vs `host_sdk_sample.h:1937` | 以代码为准 |
| D3 | README 表里 `odin1/depth_img_competetion` 拼写有误（competetion），但代码里就是这个拼写 | `depth_image_ros2_node.cpp:27-28` | 按代码拼写订阅 |
| D4 | `cloud_slam` 的 `frame_id="map"`，但 Wiki §5.4 说这些 topic 在 odom 系发布 | `host_sdk_sample.h:1935` 上下文、Wiki §5.4 | **V1 待实测** |
| D5 | `T_imu_lidar` 平移 x 值：Wiki `-0.02663` vs 代码 `+0.00347` | Wiki §5.2 vs `host_sdk_sample.h:1890` | **V2 待实测**，运行期 TF 为准 |
| D6 | README 未列出 `wiwc`、`camera_pose_visual`、`image/intensity_gray` 三个 topic | `host_sdk_sample.h:1939,1943,1944` | 本项目不用，但排障时要知道它们存在 |
| D7 | `odometry` 频率只在 Wiki §5.4 出现（10 Hz），README 未写 | Wiki §5.4 | **V5 待验证** |
| D8 | `wiwc` 的语义（world-in-world-camera?）文档未定义 | 代码 `:1243` 只有发布 | 不使用；若要用先问厂商 |

## 10. 参考

- 驱动仓库：<https://github.com/manifoldsdk/odin_ros_driver>（Apache-2.0，v0.14.4）
- 官方 Wiki 入口：<https://manifoldtechltd.github.io/wiki/odin_series/odin1/>
- 数据输出（坐标系/外参/TF）：<https://manifoldtechltd.github.io/wiki/odin_series/odin1/5.%20Data%20output_.html>
- 技术参数：<https://manifoldtechltd.github.io/wiki/odin_series/odin1/14.%20Technical%20Specifications_.html>
- 时间同步：<https://manifoldtechltd.github.io/wiki/odin_series/odin1/12.%20Time%20Synchronization.html>
