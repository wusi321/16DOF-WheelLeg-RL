"""梅花桩任务专用的奖励 / 终止 / 课程项（对应 16DOF_RL.md 第 3、5、6 节）。

设计要点：

* **进度奖励用机身实际位移**（世界系速度投影到指令方向），不用轮速，避免
  “轮子空转也算前进”的奖励漏洞。
* **轮滑惩罚**只在轮子确实接触地形时生效，避免悬空时误罚。
* **成功条件**是机身进入终点平台；成功后按“终止”处理（不是 time-out），
  值函数不会对未来做 bootstrap。
* **课程**按“是否通过当前难度”升/降级，而不是只看总回报。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .. import naming

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def _root_x_rel(env: "ManagerBasedRlEnv", asset: Entity) -> torch.Tensor:
    """机身相对本环境出生点（地形 patch 原点）的 x 位移。"""
    return asset.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]


def forward_progress(
    env: "ManagerBasedRlEnv",
    command_name: str,
    target_speed: float = 0.8,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """目标方向上的实际前进速度（归一化到 [0, 1]）。

    指令接近零时退化为 +x 方向，保证静止起步阶段也有前进信号。
    """
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    asset: Entity = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)[:, :2]
    # 指令是机体系（UniformVelocityCommand），所以速度也必须取机体系；
    # 用世界系速度去点乘机体系指令，在偏航后会把正确跟踪算成 0（mjlab 的速度跟踪
    # 奖励同样用 root_link_lin_vel_b，保持一致）。
    v = asset.data.root_link_lin_vel_b[:, :2]

    norm = torch.norm(cmd, dim=1, keepdim=True)
    fallback = torch.zeros_like(cmd)
    fallback[:, 0] = 1.0
    direction = torch.where(norm > 1.0e-3, cmd / torch.clamp(norm, min=1.0e-6), fallback)

    progress = torch.sum(v * direction, dim=1)
    return torch.clamp(progress, min=0.0) / max(target_speed, 1.0e-6)


def body_height_band(
    env: "ManagerBasedRlEnv",
    min_height: float = 0.18,
    max_height: float = 0.75,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """机身高度越界惩罚：过低（掉进桩间）或过高（异常姿态）都罚。"""
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    asset: Entity = env.scene[asset_cfg.name]
    z = asset.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return torch.clamp(min_height - z, min=0.0) + torch.clamp(z - max_height, min=0.0)


def wheel_slip(
    env: "ManagerBasedRlEnv",
    sensor_name: str,
    wheel_radius: float,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """轮地滑移：|ω·r − v_x|，只统计处于接触状态的轮子。"""
    from mjlab.sensor import ContactSensor

    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot", joint_names=naming.WHEEL_JOINT_PATTERNS)
    asset: Entity = env.scene[asset_cfg.name]
    sensor: ContactSensor = env.scene[sensor_name]

    found = (sensor.data.found > 0).float()  # [B, P]，P = 轮子数
    wheel_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    body_vx = asset.data.root_link_lin_vel_b[:, 0:1]

    n = min(found.shape[1], wheel_vel.shape[1])
    slip = torch.abs(wheel_vel[:, :n] * wheel_radius - body_vx)
    contact = found[:, :n]
    return torch.sum(slip * contact, dim=1) / torch.clamp(contact.sum(dim=1), min=1.0)


def reached_goal(
    env: "ManagerBasedRlEnv",
    distance: float,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """机身沿 +x 走过的距离超过 distance 即视为到达终点（bool 张量）。"""
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    asset: Entity = env.scene[asset_cfg.name]
    return _root_x_rel(env, asset) > distance


def goal_reached_bonus(
    env: "ManagerBasedRlEnv",
    distance: float,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """到达终点时给一次 1.0 的奖励（配合正权重使用）。"""
    return reached_goal(env, distance, asset_cfg).float()


def base_too_low(
    env: "ManagerBasedRlEnv",
    min_height: float = 0.12,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """机身高度低于阈值：掉进桩间或摔倒。"""
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    asset: Entity = env.scene[asset_cfg.name]
    z = asset.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return z < min_height


def stepstone_levels(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor,
    goal_distance: float,
    min_progress_frac: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> dict[str, torch.Tensor]:
    """梅花桩课程：通过整段地形升级，几乎原地不动降级。

    与 `terrain_levels_vel_strict` 一致，在 reset 之前调用，因此读到的位置
    是本回合结束时的位置。
    """
    asset: Entity = env.scene[asset_cfg.name]
    terrain = env.scene.terrain
    assert terrain is not None
    assert terrain.env_origins is not None

    # 用**沿 +x 的有符号位移**判定，而不是 XY 欧氏距离：成功条件（reached_goal）只看 x，
    # 欧氏距离会把横向漂移算成“前进”，导致走歪但没过的回合也被升级。
    dx = asset.data.root_link_pos_w[env_ids, 0] - terrain.env_origins[env_ids, 0]

    # 首次 reset 时机器人数据还停在世界原点，distance 是假的，会误判升级。
    # 用 episode_length_buf 过滤掉“还没跑过任何 step”的环境。
    valid = env.episode_length_buf[env_ids] > 0

    move_up = (dx > goal_distance) & valid
    move_down = (dx < goal_distance * min_progress_frac) & ~move_up & valid

    terrain.update_env_origins(env_ids, move_up, move_down)

    levels = terrain.terrain_levels.float()
    return {
        "mean": torch.mean(levels),
        "max": torch.max(levels),
        "upgraded": move_up.float().mean(),
        "downgraded": move_down.float().mean(),
        "distance": torch.mean(dx),
    }
