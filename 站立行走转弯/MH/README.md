# MH：抗干扰训练

基于本仓库 MXZ `7f97d153240c22f796c80421e77bf0a86dc0e6f2` 的 `model_1795.pt`，保留直行和原地转向能力，加入随机外力测试并继续训练。
训练日期：2026-11-29。直行和转向分别累计增训 **500 轮**，各使用 512 个并行环境。

本文件夹包含匹配的权重、机器人模型/网格、控制代码、训练所用 mjlab 源码、依赖清单、启动器和验收数据。可以单独复制整个 MH 文件夹到非 C 盘运行，不依赖上传者的 F 盘路径，也不读取其他队友目录。

## 最快运行

已验证环境：Windows、Python **3.12**、NVIDIA GPU；PyTorch **2.11.0+cu128**、MuJoCo **3.8.1**、mujoco-warp **3.8.0.2**、Warp **1.12.1**、RSL-RL **5.4.0**。

在 MH 文件夹打开 PowerShell，首次安装：

```powershell
# python 命令需要对应 Python 3.12；也可以 -Python 指定其完整路径
.\setup_env.ps1
# 打开实时 MuJoCo 窗口，默认已开启 40N、0.2 秒随机推力
.\run.ps1
```

已经有兼容环境的队友可以直接指定解释器，不重复安装：

```powershell
$env:MH_PYTHON = 'D:\你的已有环境\.venv\Scripts\python.exe'
.\run.ps1 verify
.\run.ps1 view
```

启动器会使用本文件夹内的框架/机器人源码。运行输出留在 MH 内；若路径全英文，缓存也留在 MH/.cache。
本仓库的“站立行走转弯/MH”含中文，Windows CUDA首次编译不支持中文缓存路径，因此启动器会自动把缓存放到同一仓库根目录的 `.cache/MH`，不会写到C盘。
如果整个仓库的路径也含中文，请把整个MH单独复制到例如 `D:\robot\MH`，或设置 `$env:MH_CACHE_ROOT` 为本项目内的非C盘英文缓存目录。
`setup_env.ps1` 不会下载 Python 本体；请先准备 Python 3.12 和可用的 NVIDIA 驱动。依赖安装需要联网。

| 按键 | 功能 |
| --- | --- |
| W / S | 前进速度增加/减小，0～0.8m/s |
| A / D | 左/右原地转向，最大 ±0.8rad/s |
| X | 停车 |
| T | 自动演示：直行、左右转、停车循环 |
| B | 随机推力开关 |
| F | 切换 20/40/60N 推力 |
| U | 切换 0.1/0.2/0.3 秒持续时间 |
| G | 立即触发随机方向推力 |
| Enter / 空格 | 复位 / 暂停 |

鼠标旋转、缩放沿用 MuJoCo 原生操作。界面红色箭头显示推力，左上角显示指令、推力大小、持续时间及累计摔倒次数。

```powershell
# 从手动模式开始
.\run.ps1 view --manual
# 无窗口检查完整的30秒运动循环
.\run.ps1 view --headless --steps 1500
# 使用原 model_1795 对照，不加载这次新训练权重
.\run.ps1 view --original
# 写演示视频到本目录 logs/，需支持图形渲染
.\run.ps1 view --video logs/demo.mp4 --steps 1500
```

## 权重和控制契约

| 文件 | 含义 | 使用方式 |
| --- | --- | --- |
| `checkpoints/straight.pt` | 直行残差策略，200+300轮 | 在固定腿参考 PD 和轮速前馈上叠加有界修正 |
| `checkpoints/turn.pt` | 原转向 actor/critic 微调，200+300轮 | 转向时轮速目标为0，由腿部步态完成转向 |
| `vendor/mxz/checkpoints/model_1795.pt` | 未修改的原权重 | 对照以及模型来源核对 |

**请使用本目录的 run.ps1 / tools/mxz_robust.py 加载。** 直行残差权重不能直接塞到旧回放脚本中当成完整位置动作策略。
停车继续使用已通过抗推测试的原零腿动作 PD。直行只在 `abs(vx)>=0.05m/s` 时启用残差；最大修正为 hip ±0.08rad、thigh/calf ±0.12rad、轮速 ±2rad/s，随后仍做物理目标限幅。
两份 actor 都是原来的 **53维本体观测**，不读取真实机身线速度或外力作为 actor 输入。
本控制契约覆盖前进、停车和原地左右转向，不包含已经验证的弧线、侧移或后退。

- 软件零点 `zero-45-90-v1`：前腿 0°/+45°/−90°，后腿 0°/−45°/+90°。
- 小腿绝对角限位：FR/FL **[-2.7,-1.21]rad**；RL/RR **[1.21,2.7]rad**；物理关节、滤波后动作目标及重建执行器同时限幅。
- 电机力矩配置：hip/thigh **40N·m**、calf **60N·m**、wheel **15N·m**。不能与其他模型的 30/20/5 配置混用成绩。
- 物理步长 **0.005s**，decimation **4**，控制频率 **50Hz**。源 MJCF 中的 option/初始高度由机器人装配和环境配置覆盖，运行契约以环境为准。
- 保留原 MXZ 的碰撞体修正和惯量；软件参考角不是实机编码器已标定的证明。

## 干扰与训练

使用真实外力接口：世界坐标水平面 **8方向**，作用点为机身质心沿机身z轴上方 **8cm**，力矩为 `r × F`。
默认推力持续 **0.2秒**，先稳定5秒，再在每次推力结束后间隔 **5～8秒** 随机施加。没有用瞬间改速度代替外力。
这次训练保持20N，未在前档未通过时提高强度；两次训练均用 `1e-4` 学习率，每阶段重新建立优化器。
保留观测噪声、0～2控制步延迟、0.8～1.2轮子摩擦倍率；25%的episode不加推力。
直行收紧速度/姿态/高度奖励；转向收紧偏航跟踪，并针对右转漂移收紧x/y平移跟踪、增加左右 ±0.8rad/s 样本。
各阶段的实际配置、TensorBoard事件及训练时源码快照位于 `training/`。历史快照用于追溯，运行入口始终是 `tools/mxz_robust.py`。

## 验收结果

在相同的 **MuJoCo无限平面** 上比较原模型和新模型。六个场景：停车、0.4/0.8m/s直行、±0.8rad/s原地转向、每6秒切换一次模式。
每场景每档推力使用32个环境×3个种子×30秒，含无推力对照和20/40/60N测试；总计 **2304段仿真、6822次推力事件**。

| 40N、0.2秒 | 原模型恢复率 | 新模型恢复率 |
| --- | ---: | ---: |
| 停车 / 0.4直行 / 0.8直行 | 各100% | 各100% |
| 左转 | 29.3% | 100% |
| 右转 | 0% | 100% |
| 模式切换 | 74.1% | 100% |

恢复要求：推力结束后2秒内，roll/pitch各<5°、平面速度向量误差<0.08m/s、偏航速度误差<0.15rad/s，并连续保持0.5秒。
原模型低恢复率表示未连续满足这个严格门槛，不等于不能转向，也不等于摔倒。
新模型全部测试未摔倒、无NaN、动作目标无越界，实测小腿限位穿透0；无推力速度跟踪也通过回归检查。
20/40N为正式验收，60N单列为压力测试。结论限于这些平地仿真工况，不能作为任意扰动、崎岖地面或实机保证。

纯直行/停车沿用同一份直行权重的已完成数据；左右转向和模式切换用最终两份权重重新测。合并前核对直行SHA256、推理/外力代码AST、限位、地面和矩阵键，逐行来源见结果文件。
早期有限方块场地会在长距离测试中驶出边缘，其数据未混入正式对照。

- [详细验收报告](results/final/summary.md)
- [原模型与新模型对比图](results/final/before_after.png)
- [各力档恢复/速度曲线](results/final/robustness_curves.png)
- [40N模式切换时序](results/final/switch_trace_40N.png)
- [训练曲线](training/training_curves.png)
- `results/final/results.json`、`summary.json`、逐次推力记录和 `results/baseline/` 保存结构化统计。
- `MANIFEST.json` 记录模型身份与控制契约；`SHA256SUMS` 可通过 `.\run.ps1 verify` 校验整个交付。

## 继续评测 / 训练

```powershell
.\run.ps1 evaluate --straight checkpoints/straight.pt --turn checkpoints/turn.pt --output logs/eval_my_machine
# 固定种子、推力档位用带引号的逗号字符串，例如 --seeds '42,43,44' --forces '0,20,40,60'

# 在这次直行残差上继续训练；务必另开输出目录
.\run.ps1 train --mode straight --force 20 --iterations 200 --checkpoint checkpoints/straight.pt --output logs/straight_continue
# 或在转向权重上继续训练
.\run.ps1 train --mode turn --force 20 --iterations 200 --checkpoint checkpoints/turn.pt --output logs/turn_continue
```

原权重不会被续训覆盖。默认入口仍加载已验收的 `checkpoints/selection.json`；候选必须先与无推力基线比较再启用。
正式统计图不是实机成绩。复现时使用本目录的框架源码；本地PyPI同版本不一定包含相同的环境接口。

## 文件来源

MXZ机器人代码和模型来自本仓库上述提交，原作者目录未修改。`vendor/mjlab` 是本机实际训练使用的1.4.0源码，保留其 Apache-2.0 LICENSE 和来源信息。
为使依赖安装与实测一致，交付中的 mjlab 安装元数据将 mujoco-warp 从原声明的 `>=3.8.0.3` 对齐为实际使用的 `3.8.0.2`；框架运行源码保持训练时版本。
未包含虚拟环境、CUDA缓存、临时目录或中间训练权重。再次运行的日志保存在本目录 `logs/`，缓存位置按上面的英文路径规则选择。
