# 02 · 统一感知接口设计

目标：让 **MuJoCo 仿真** 与 **真实 Odin1** 在“给策略看的东西”这一层完全同构，
同时保证 **训练进程永远不 import ROS 2、也不需要连着设备**。

## 1. 设计原则

| # | 原则 | 落地方式 |
| --- | --- | --- |
| P1 | 训练进程与传感器解耦 | 训练只消费 `TerrainFrame`（numpy/torch 数组）；ROS 2 只在独立进程里出现 |
| P2 | 语义唯一 | 高度、坐标系、无效值、尺寸全部由 `TerrainFrame` 定义，仿真/真机各自负责填同一套字段 |
| P3 | 策略与全局地图无关 | 观测一律在 `base_link` 系；`map`/`odom` 只用于导航层（见 04 文档 §7） |
| P4 | 可回放、可回归 | 真实数据落成 `.npz` 帧包，离线回放无需 ROS；仿真侧用同样的帧包格式做单测 |
| P5 | 失败要显式 | 无数据 / 超时 / 全无效 / NaN → 明确的 `status`，绝不静默给零 |

## 2. 进程与数据流

```text
┌────────────────────── 训练机（无 ROS，无设备） ──────────────────────┐
│  mjlab env ──raycast──► SimTerrainSource ─┐                          │
│                                           ├─► TerrainFrame ─► ObsEncoder ─► policy (273 维)
│  .npz 回放 ──────────► ReplayTerrainSource ┘                          │
└──────────────────────────────────────────────────────────────────────┘

┌────────────────────── 机器人机载（有 ROS 2） ────────────────────────┐
│  odin_ros_driver ──/odin1/*──► PerceptionNode ──► TerrainFrame ──► ShmRing/UDP ──► 策略进程
│  （只在这个进程里 import rclpy）        （硬件无关的纯 numpy 计算）
└──────────────────────────────────────────────────────────────────────┘
```

关键点：

- **仿真侧**：`SimTerrainSource` 直接读 mjlab 的 `RayCastSensor`，不经过网络、不经过 ROS。
- **真机侧**：`PerceptionNode` 是唯一知道 ROS 的存在；它输出的是与仿真同构的 `TerrainFrame`。
- **训练进程**：只认识 `TerrainFrame` 的三种实现（内存、共享内存环、文件回放），
  依赖只有 `numpy`/`torch`（仓库已有），不需要 `rclpy`。
- **策略进程（第三阶段）**：只读共享内存里的最新帧；ROS 挂了它仍能按 `status` 降级。

## 3. 数据契约：`TerrainFrame`

```python
# 逻辑定义，不是可运行代码；字段名就是接口名
SCHEMA_VERSION = 1

@dataclass
class TerrainFrame:
    schema_version: int          # = 1，布局变更时必须 +1
    stamp_ns: int                # 该帧点云采集时刻（机器人单调时钟，纳秒）
    status: str                  # "ok" | "stale" | "no_data" | "all_invalid" | "error"
    # 栅格几何（每帧携带，允许平滑升级时不一致）
    rows: int                    # 13
    cols: int                    # 21
    resolution_m: float          # 0.08
    # 观测主体：base_link 系（yaw 对齐），单位米，正值 = 该点在地面上方多少
    height_m: np.ndarray         # float32[rows, cols]
    valid: np.ndarray            # bool[rows, cols]
    confidence: np.ndarray       # float32[rows, cols], 0..1，供后续加权/训练增强
    # 可选：仅供调试/可视化，不进策略
    debug: dict | None = None
```

字段语义（**与现有仿真观测逐项对齐**，见 04 文档）：

| 字段 | 定义 | 无效时 |
| --- | --- | --- |
| `height_m[r, c]` | `z_base − z_ground`：基座系下“地面点比基座低多少”，单位 m。仿真里就是 `frame_z − hit_z`；真机里就是 `−(T_base←world · p).z` | 任意有限值 + `valid=False`（编码器负责替换） |
| `valid[r, c]` | 该格有点 / 有命中 | — |
| `confidence[r, c]` | 归一化置信度（仿真可全 1.0） | 0 |
| `stamp_ns` | **采集时刻**，不是发布时刻 | — |
| `status` | 整帧状态；策略侧只认 `ok` | — |

栅格布局（已用探针在本仓库验证）：

- 行 `r`（0..12）对应 **y**，列 `c`（0..20）对应 **x**；
- x ∈ [−0.80, 0.80] m（21 点），y ∈ [−0.50, 0.46] m（13 点，**y 不对称，偏左 2 cm**）；
- 展平顺序为 **x 连续**（`index = r*21 + c`），与 `GridPatternCfg.generate_rays` 输出一致；
- 射线方向在 `base_link` 系为 `(0, 0, −1)`，`ray_alignment="yaw"`（只跟随偏航，不跟随俯仰/横滚）。

## 4. 三个数据源实现

### 4.1 `SimTerrainSource`（仿真）

- 输入：`env.scene["height_scanner"]` 的 `distances / hit_pos_w / frame_pos_w`。
- 计算：`h = frame_z − hit_z`；`valid = distances >= 0`。
- 与现有 `height_scan` 完全一致（`mjlab/src/mjlab/envs/mdp/observations.py:126-163`）。
- 额外能力（仅训练用，真机可选）：随机丢点、噪声、按距离的稀疏化，用于 sim2real 增强（见 03 文档）。

### 4.2 `Odin1TerrainSource`（真机 / 离线 bag）

输入：`/odin1/cloud_raw`（`PointCloud2`，字段 `x,y,z,intensity,confidence,offset_time`）、
`/odin1/imu`（400 Hz）、`/odin1/odometry_highfreq`（400 Hz）、TF `odom→imu`、`imu→lidar`。

处理链（每一步都可单独单测）：

1. **去畸变**：按 `offset_time` 用 IMU/里程计把点变换到同一时刻（实现前先做 V3 验证）。
2. **过滤**：`confidence >= threshold`（默认 35）；`r ∈ [0.2, max_range]`；剔除 NaN/Inf。
3. **变换**：`p_base = T_base←imu · T_imu←lidar · p_lidar`，`T_base←imu` 由 TF 取当前位姿**并去滚转/俯仰**
   （只保留 yaw，和仿真的 `ray_alignment="yaw"` 对齐）。
4. **栅格化**：按 x/y 落到 21×13 的格子；每格取**下分位数**（默认 20%，稳健于飞点），
   同时记录点数与置信度均值。
5. **补洞**：只做“**有有效邻居的洞**”的一次 3×3 邻域填补；边缘外的洞保持 `valid=False`。
6. **状态**：所有格子都无效 → `all_invalid`；帧龄超阈值 → `stale`（见 §6）。

### 4.3 `ReplayTerrainSource`（离线验证 / 仿真单测）

- 文件格式：`.npz`，键 `height_m, valid, confidence, stamp_ns, rows, cols, resolution_m, meta`（`meta` 为 JSON 字符串）。
- 一个 bag → 一个 npz（`stamp_ns` 数组 + 每帧堆叠），或每帧一个文件；帧数 ≤ 数万，npz 足够。
- 好处：**第二阶段验证不需要 ROS、不需要设备**，CI 也能跑。

## 5. 观测编码器（唯一允许“变形”的地方）

```python
# 与现有仿真观测逐位对齐，任何改动必须走新 task id（例如 Robot-StepStone-v2）
MISS_VALUE = 1.0          # 现有 clip(-1, 1) 下，未命中 5.0 → 1.0
CLIP = (-1.0, 1.0)

def encode(frame: TerrainFrame) -> np.ndarray:      # -> float32[273]
    h = np.where(frame.valid, frame.height_m, 5.0)  # 未命中先按 5 m
    return np.clip(h, *CLIP).reshape(-1)            # 再截断
```

规则：

- `valid=False` → `1.0`（与现有策略训练时的输入**完全一致**）；
- 取值范围 `[−1, 1]`，超出按边界截断（坑深 > 1 m 或地面高于基座 1 m 时都是边界值）；
- 编码器只负责“`TerrainFrame` → 273 维”，**不做任何滤波**；滤波在数据源里做，便于复现。

> 现有策略的 actor 输入是 53 维本体 + 273 维高度（StepStone `Robot-StepStone-v0`，actor 326 维）。
> 编码器的输出必须**逐元素**等于 `safe_height_scan` 的结果，这条在第 1 阶段用黄金样本验证。

## 6. 时序、延迟与降级

| 场景 | 策略 |
| --- | --- |
| 仿真 | 每步同步取一帧，`stamp_ns = sim_time`，无延迟 |
| 真机 | 点云 10 Hz，策略 50 Hz：**零阶保持**最新帧，最多保持 `max_hold_ms`（建议 200 ms，约 1 帧余量 + 处理时间） |
| 超时（`now − stamp_ns > max_age_ms`，建议 300 ms） | `status="stale"`，`valid` 全 False → 观测恒为 1.0，策略按“看不见”的分布退化（训练时要有对应增强） |
| 无数据（首次启动/驱动重启） | `status="no_data"`，同上 |
| 单帧部分无效 | 保持该帧，`valid` 如实反映；不整帧丢弃 |
| 时钟跳变 | 只用**单调时钟**做帧龄；`stamp_ns` 若是设备时间（`use_host_ros_time=0`），入队时立即用单调时钟记 `recv_ns` 辅助 |

设计决定：**不插值**。地形高度图插值会在桩边缘制造假台阶；宁可零阶保持 + 明确的 `stale`。

## 7. 计算放在哪一侧

| 计算 | 位置 | 理由 |
| --- | --- | --- |
| TF 查询、点云反投影、栅格化 | `PerceptionNode`（机载，可能用 C++/Python + numpy） | 只依赖基础数学，和控制解耦 |
| 观测截断/展平 | 策略侧（5 行代码） | 保证“策略看到的就是训练时看到的” |
| 随机化（噪声/丢点） | **训练侧**（数据源包装） | 真机不该加噪声 |
| 安装外参标定 | 配置（YAML），不入代码 | 换装不重编 |

配置示例（`perception.yaml`，真机侧）：

```yaml
grid: {rows: 13, cols: 21, resolution_m: 0.08}
mount:            # V4 待实测，先留空
  translation_m: [0.0, 0.0, 0.0]
  rpy_deg: [0.0, 0.0, 0.0]
cloud:
  topic: /odin1/cloud_raw
  confidence_min: 35
  range_m: [0.2, 30.0]
  cell_percentile: 0.2
time:
  max_age_ms: 300
  max_hold_ms: 200
```

## 8. 失败与边界处理清单

| 情况 | 行为 | 可观测证据 |
| --- | --- | --- |
| `PointCloud2` 字段缺 `confidence` | 记 `error` 一次，按无置信度处理（不过滤），继续跑 | `debug["missing_fields"]` |
| 点全被过滤 | `all_invalid` + `valid=False` | 计数 |
| 点云时间戳倒退 | 丢弃该帧，计 `clock_backward` | 日志 |
| TF 不可用 | 用 400 Hz 里程计位姿兜底；再无 → `error` | `debug["tf_fallback"]` |
| NaN/Inf | 该点丢弃；整帧 NaN → `error` | 计数 |
| 点数 < 阈值（如 500） | 仍生成帧，`confidence` 压低 | `debug["n_points"]` |

## 9. 版本与兼容

- `SCHEMA_VERSION` 变更 → 训练侧拒绝加载旧 npz（防止“以为对齐了其实没有”）。
- 观测布局用 `OBS_LAYOUT_ID = "hscan13x21x0.08+yaw+clip1+v1"` 记录在 checkpoint 的 `extras` 里；
  换布局必须换 task id，旧 checkpoint 不允许续训。
- 现有 `Robot-StepStone-v0` 的 273 维语义**冻结**，后续增强走 `-v2`。

## 10. 验收（本阶段只到“设计可落地”）

- [ ] 文档层面的字段/单位/坐标约定无歧义（本文 + 04 文档）。
- [ ] 三个数据源都能产出同一 `TerrainFrame`（第 1 阶段实现仿真 + 回放两个；真机第 3 阶段）。
- [ ] 编码器与现有 `safe_height_scan` 在 20 个随机仿真状态上逐元素一致（黄金测试）。
- [ ] 训练进程在不安装 `rclpy`、无设备的环境下可完整跑通（CI 已能证明：现有冒烟测试不依赖 ROS）。
