"""速度指令采样器：专门为「直行 + 步态转向」设计。

mjlab 自带的 ``UniformVelocityCommand`` 在 (vx, vy, wz) 的整个盒子里均匀采样，
大部分样本都是「斜着走 + 边转边走」的混合动作，纯直行和纯原地转向的样本很少。
本任务只关心两件事：

* **直行**（vx 有值，vy = wz = 0）
* **转向**（原地转向，或边走边转的弧线）

所以这里把每个环境的指令按权重分成 5 类：

================  ======  ==================================================
模式              权重    vx / vy / wz
================  ======  ==================================================
直行 straight      0.45   vx ∈ ranges.lin_vel_x（含倒车），vy = 0，wz = 0
原地转向 yaw       0.20   vx = 0，vy = 0，|wz| ≥ min_abs_turn
弧线 arc           0.20   vx ≥ min_abs_lin（只向前），wz ≠ 0
横移 lateral       0.10   vx = 0，vy ∈ ranges.lin_vel_y，wz = 0
站立 standing      ``rel_standing_envs``（默认 0.05，整条指令为 0）
================  ======  ==================================================

权重会自动归一化；站立比例用的是基类的 ``rel_standing_envs``，不占上面四个权重。

另外每个轴的采样都做了「死区」处理：|vx| 或 |wz| 不会落在 0 附近那种
「指令几乎为 0 但又不是 0」的尴尬区间，避免策略学到又不动又不敢动的行为。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.tasks.velocity.mdp.velocity_command import (
    UniformVelocityCommand,
    UniformVelocityCommandCfg,
)

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def _sample_range(n: int, bounds: tuple[float, float], device: torch.device) -> torch.Tensor:
    lo, hi = bounds
    return torch.empty(n, device=device).uniform_(lo, hi)


def _deadzone(value: torch.Tensor, min_abs: float) -> torch.Tensor:
    """把接近 0 的采样值推到 ±min_abs，保留原来的正负号（0 视为正）。"""
    sign = torch.where(value < 0, -torch.ones_like(value), torch.ones_like(value))
    return sign * value.abs().clamp(min=min_abs)


class StraightTurnVelocityCommand(UniformVelocityCommand):
    """按「直行 / 原地转向 / 弧线 / 横移 / 站立」分类采样的速度指令。"""

    cfg: StraightTurnVelocityCommandCfg

    def __init__(self, cfg: StraightTurnVelocityCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        #: 最近一次采样的模式（0=直行 1=原地转向 2=弧线 3=横移 4=站立），调试用。
        self.command_mode = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        weights = torch.tensor(
            [
                cfg.fraction_straight,
                cfg.fraction_turn_in_place,
                cfg.fraction_arc,
                cfg.fraction_lateral,
            ],
            device=self.device,
        )
        if float(weights.sum()) <= 0.0:
            raise ValueError("StraightTurnVelocityCommandCfg 的模式权重之和必须大于 0")
        # 累积分布的分割点，用于 torch.bucketize（0=直行 1=原地转向 2=弧线 3=横移）。
        self._mode_weights = weights / weights.sum()
        self._mode_edges = torch.cumsum(self._mode_weights, dim=0)[:-1]

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        if n == 0:
            return
        device = self.device
        cfg = self.cfg
        cmd = self.vel_command_b[env_ids]

        # 1) 先决定每个环境的模式（站立单独按 rel_standing_envs 采样）。
        standing = torch.rand(n, device=device) < cfg.rel_standing_envs
        mode = torch.bucketize(torch.rand(n, device=device), self._mode_edges)
        mode = torch.where(standing, torch.full_like(mode, 4), mode)

        # 2) 三个轴各自在指令范围内采样（后面按模式屏蔽）。
        vx = _sample_range(n, cfg.ranges.lin_vel_x, device)
        vy = _sample_range(n, cfg.ranges.lin_vel_y, device)
        wz = _sample_range(n, cfg.ranges.ang_vel_z, device)

        straight = mode == 0
        turn = mode == 1
        arc = mode == 2
        lateral = mode == 3
        still = mode == 4

        vx = torch.where(straight, _deadzone(vx, cfg.min_abs_lin), vx)
        vx = torch.where(arc, vx.abs().clamp(min=cfg.min_abs_lin), vx)
        vx = torch.where(turn | lateral | still, torch.zeros_like(vx), vx)

        vy = torch.where(lateral, _deadzone(vy, cfg.min_abs_lin), torch.zeros_like(vy))
        vy = torch.where(still | turn | straight | arc, torch.zeros_like(vy), vy)

        wz = torch.where(turn | arc, _deadzone(wz, cfg.min_abs_turn), torch.zeros_like(wz))
        wz = torch.where(still | straight | lateral, torch.zeros_like(wz), wz)

        cmd[:, 0] = vx
        cmd[:, 1] = vy
        cmd[:, 2] = wz
        self.vel_command_b[env_ids] = cmd
        self.vel_command_w[env_ids] = cmd

        self.is_standing_env[env_ids] = still
        self.is_heading_env[env_ids] = False
        self.is_world_env[env_ids] = False
        self.is_forward_env[env_ids] = straight
        self.command_mode[env_ids] = mode


@dataclass(kw_only=True)
class StraightTurnVelocityCommandCfg(UniformVelocityCommandCfg):
    """直行 + 转向指令的采样权重。权重会自动归一化。"""

    class_type: type = StraightTurnVelocityCommand

    #: v2（转向优先）：把样本重心从「直行」挪到「转向/弧线」——
    #: pilot1 里 straight 占 45%，策略只学滚轮就够了。
    fraction_straight: float = 0.25
    fraction_turn_in_place: float = 0.35
    fraction_arc: float = 0.25
    fraction_lateral: float = 0.15

    #: 线速度死区（m/s）：直行时 |vx| 不低于它，横移时 |vy| 不低于它。
    min_abs_lin: float = 0.2
    #: 角速度死区（rad/s）：转向指令 |wz| 不低于它，避免「想转但几乎不转」。
    min_abs_turn: float = 0.3

    def build(self, env: ManagerBasedRlEnv) -> StraightTurnVelocityCommand:
        """注意：必须重写 build()。父类 ``UniformVelocityCommandCfg.build()`` 写死了
        返回 ``UniformVelocityCommand``，只改 ``class_type`` 不会生效。"""
        return StraightTurnVelocityCommand(self, env)