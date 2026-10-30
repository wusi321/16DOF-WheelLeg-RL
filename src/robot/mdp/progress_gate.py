"""防止"原地小跳/原地磨时间"骗奖励的通用项。

2026-10-30 的 play 里，策略学会在原地反复小跳（靠落地次数刷 `leap` 奖励），
几乎不前进。`leap` 本身已经加了"必须真的前进 >= min_advance"的门槛，
这里再加一条**通用**的兜底：在赛道内连续 `idle_steps` 步没有前进
（最好位置 x 没有刷新），就按步罚分。

它和具体任务是解耦的：沟壑、绕桩、梅花桩都可以挂，参数只决定"多久算磨蹭"。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg


@dataclass
class _IdleState:
    best_x: torch.Tensor | None = None
    idle: torch.Tensor | None = None
    last_buf: torch.Tensor | None = None
    last_out: torch.Tensor | None = None


def _state(env) -> _IdleState:
    state = getattr(env, "_idle_state", None)
    if state is None:
        state = _IdleState()
        env._idle_state = state
    if state.best_x is None:
        state.best_x = torch.full((env.num_envs,), -1e3, device=env.device)
        state.idle = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    return state


def _step(
    env,
    idle_steps: int,
    min_gain: float,
    x_min: float,
    x_max: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    state = _state(env)
    buf = env.episode_length_buf.to(torch.long)
    if state.last_buf is not None and torch.equal(buf, state.last_buf):
        return state.last_out  # type: ignore[return-value]

    reset = (buf <= 1) if state.last_buf is None else ((buf < state.last_buf) | (buf <= 1))
    if bool(reset.any()):
        state.best_x[reset] = -1e3
        state.idle[reset] = 0

    asset: Entity = env.scene[asset_cfg.name]
    x = asset.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]

    gained = x > state.best_x + float(min_gain)
    state.best_x = torch.where(gained, x, state.best_x)
    state.idle = torch.where(gained, torch.zeros_like(state.idle), state.idle + 1)

    in_course = (x >= float(x_min)) & (x <= float(x_max))
    out = in_course & (state.idle >= int(idle_steps))
    state.last_buf = buf.clone()
    state.last_out = out
    return out


def idle_penalty(
    env,
    idle_steps: int = 75,
    min_gain: float = 0.03,
    x_min: float = 0.3,
    x_max: float = 7.2,
    scale: float = -0.5,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """赛道内连续 `idle_steps` 步没有前进 -> `scale`/步。"""
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    out = _step(env, idle_steps, min_gain, x_min, x_max, asset_cfg)
    return out.to(torch.float32) * scale
