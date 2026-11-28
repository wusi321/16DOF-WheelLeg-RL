"""16DOF-WheelLeg 样机的命名、几何与标称姿态常量。

本模块集中所有「跟具体模型有关」的名字和数值。换模型时只改这里和
``robot_cfg.py``，上层环境配置不用动。

数据来源：``16DOF-URDF/mujoco/16DOF-URDF-training.xml``（URDF 导出的
box/cylinder 碰撞训练模型）。所有数值都用 ``tools/probe_*.py`` 在本机实测，
不要凭空改。

模型速览
--------
* 整机实测质量 25.1 kg（旧文档写的 27.1 kg 是算术错误，不影响增益）。
* 四条腿：髋绕 x（外摆）、大腿绕 y、小腿绕 y、轮子绕 y（无角度限位）。
* **前后腿的小腿符号相反**：前腿 calf ∈ [-2.7, -0.9]，后腿 calf ∈ [+0.9, +2.7]；
  髋范围也前后镜像（FR/RR: -0.9..0.7，FL/RL: -0.7..0.9）。
* 轮子半径 0.0839 m；**轮子角速度取正 = 机器人朝 +x 前进**。
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# 命名
# ---------------------------------------------------------------------------
#: 机身 body 名（MJCF 根 body 叫 base）。
BASE_BODY = "base"

#: 关节顺序：每条腿 4 个关节。
LEG_ORDER = ("FR", "FL", "RL", "RR")
FRONT_LEGS = ("FR", "FL")
REAR_LEGS = ("RL", "RR")

#: 关节正则（用于 mjlab 的 target_names_expr / 奖励里的 joint 匹配）。
HIP_JOINT_PATTERNS = (".*_hip_joint",)
THIGH_JOINT_PATTERNS = (".*_thigh_joint",)
CALF_JOINT_PATTERNS = (".*_calf_joint",)
WHEEL_JOINT_PATTERNS = (".*_wheel_joint",)
LEG_JOINT_PATTERNS = (
    ".*_hip_joint",
    ".*_thigh_joint",
    ".*_calf_joint",
)
SAGITTAL_JOINT_PATTERNS = (".*_thigh_joint", ".*_calf_joint")

#: 具体关节名（按 LEG_ORDER × hip/thigh/calf 排列），tools 和调试用。
LEG_JOINT_NAMES = tuple(f"{leg}_{part}_joint" for leg in LEG_ORDER for part in ("hip", "thigh", "calf"))
WHEEL_JOINT_NAMES = tuple(f"{leg}_wheel_joint" for leg in LEG_ORDER)
ALL_JOINT_NAMES = LEG_JOINT_NAMES + WHEEL_JOINT_NAMES

#: 不允许撞地面的腿部 body（用于 body_collision 惩罚/接触传感器）。
BODY_COLLISION_PATTERNS = (".*_hip_Link", ".*_thigh_Link", ".*_calf_Link")
#: 四个轮子 body，顺序与 LEG_ORDER 一致。
WHEEL_BODY_PATTERNS = tuple(f"{leg}_wheel_Link" for leg in LEG_ORDER)

#: 对角线镜像（步态整形）：FR<->RL，FL<->RR。
MIRROR_SAGITTAL_PAIRS = (
    ("FR_(thigh|calf)_joint", "RL_(thigh|calf)_joint"),
    ("FL_(thigh|calf)_joint", "RR_(thigh|calf)_joint"),
)
#: 同侧前后髋镜像：FR<->RR，FL<->RL。
MIRROR_ABDUCTION_PAIRS = (
    ("FR_hip_joint", "RR_hip_joint"),
    ("FL_hip_joint", "RL_hip_joint"),
)

# ---------------------------------------------------------------------------
# 几何（实测，m）
# ---------------------------------------------------------------------------
WHEEL_RADIUS = 0.0839
#: 左右轮接地点横向间距（|y|≈0.2061）。
WHEEL_TRACK = 0.4122
#: 标称姿态下的纵向轴距（前轮 x≈+0.10、后轮 x≈-0.43）。
WHEEL_BASE = 0.60

# ---------------------------------------------------------------------------
# 标称站姿
# ---------------------------------------------------------------------------
#: 实机标定：地面到 base_link 原点 33.64 cm，到模型顶端 48.24 cm。
#: 前腿 thigh +45° / calf -90°；后腿 thigh -45° / calf +90°；hip 0°。
STAND_JOINT_POS = {
    "(FR|FL)_hip_joint": 0.0,
    "(FR|FL)_thigh_joint": 0.7853981634,
    "(FR|FL)_calf_joint": -1.5707963268,
    "(RL|RR)_hip_joint": 0.0,
    "(RL|RR)_thigh_joint": -0.7853981634,
    "(RL|RR)_calf_joint": 1.5707963268,
    ".*_wheel_joint": 0.0,
}
#: 正运动学给出的 base 原点高度（轮子接地）。
STAND_BASE_HEIGHT = 0.3362
#: base_height 奖励目标：Step 1 门禁实测零动作 4 s 落点 0.3208（PD 稳态误差
#: 约 1.5 cm，关节误差约 0.054 rad），取 0.320 作为奖励目标。
STAND_TARGET_HEIGHT = 0.320

#: 具体关节名的目标角（由 STAND_JOINT_POS 展开，tools 设 qpos 用）。
STAND_JOINT_POS_BY_NAME: dict[str, float] = {}
for _leg in LEG_ORDER:
    _front = _leg in FRONT_LEGS
    STAND_JOINT_POS_BY_NAME[f"{_leg}_hip_joint"] = 0.0
    STAND_JOINT_POS_BY_NAME[f"{_leg}_thigh_joint"] = 0.7853981634 if _front else -0.7853981634
    STAND_JOINT_POS_BY_NAME[f"{_leg}_calf_joint"] = -1.5707963268 if _front else 1.5707963268
    STAND_JOINT_POS_BY_NAME[f"{_leg}_wheel_joint"] = 0.0

# ---------------------------------------------------------------------------
# 轮子方向约定
# ---------------------------------------------------------------------------
#: +wz（左转）时左侧轮（FL/RL）转得更快；+角速度 = 前进。
WHEEL_SIDE_SIGNS = (-1.0, 1.0, -1.0, 1.0)  # FR, FL, RL, RR

# ---------------------------------------------------------------------------
# 执行器标定（仿真用；真机 RS03 额定约 20 Nm，这里留余量）
# ---------------------------------------------------------------------------
EFFORT_HIP = 40.0
EFFORT_THIGH = 40.0
EFFORT_CALF = 60.0
EFFORT_WHEEL = 15.0

STIFFNESS_LEG = 80.0
DAMPING_LEG = 3.5
STIFFNESS_CALF = 150.0
DAMPING_CALF = 3.5
DAMPING_WHEEL = 1.0

# 防御：展开后的目标必须覆盖 16 个关节。
if len(STAND_JOINT_POS_BY_NAME) != 16:  # pragma: no cover
    raise RuntimeError(f"expected 16 stand joints, got {len(STAND_JOINT_POS_BY_NAME)}")

_JOINT_RE = re.compile(r"^(FR|FL|RL|RR)_(hip|thigh|calf|wheel)_joint$")
assert all(_JOINT_RE.match(name) for name in ALL_JOINT_NAMES)