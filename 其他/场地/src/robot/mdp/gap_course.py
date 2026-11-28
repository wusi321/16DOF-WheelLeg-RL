"""沟壑课程（gap course）的 MDP 项：只奖励**真的跨过沟**的跃起。

踩过的坑（2026-10-30 play）：最初只要"离地 >=2 步后落地"就给 +5，
策略立刻学会**在原地小幅反复弹跳**白拿奖励（落地就算一次），一步都不往前走。

所以现在还要求：
1. **落点比起跳点前进 >= `min_advance`**（默认 0.25 m）——原地跳不算；
2. 起跳点/落点都要落在赛道范围内（起点平台之后、终点之前）。

metric `gaps_crossed` 用同一套判定，所以它也能直接反映"是不是在真跳"。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .stepstone import wheel_contact_any


@dataclass
class _LeapState:
    air: torch.Tensor | None = None            # 截至上一刻连续离地的步数
    leaps: torch.Tensor | None = None          # 本回合有效跃沟次数
    prev_contact: torch.Tensor | None = None   # 上一刻是否有轮子接触
    takeoff_x: torch.Tensor | None = None      # 上次离地时的 x（env 相对）
    last_buf: torch.Tensor | None = None
    last_new: torch.Tensor | None = None


def _state(env) -> _LeapState:
    state = getattr(env, "_gap_leap_state", None)
    if state is None:
        state = _LeapState()
        env._gap_leap_state = state
    if state.air is None:
        state.air = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        state.leaps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        state.prev_contact = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        state.takeoff_x = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    return state


def _step(
    env,
    min_air_steps: int,
    min_advance: float,
    course_x_min: float,
    course_x_max: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """每个环境步只更新一次，返回"这一步刚刚完成一次**有效**跃沟"的 mask。"""
    state = _state(env)
    buf = env.episode_length_buf.to(torch.long)
    if state.last_buf is not None and torch.equal(buf, state.last_buf):
        return state.last_new  # type: ignore[return-value]

    reset = (buf <= 1) if state.last_buf is None else ((buf < state.last_buf) | (buf <= 1))
    if bool(reset.any()):
        state.air[reset] = 0
        state.leaps[reset] = 0
        state.prev_contact[reset] = True
        state.takeoff_x[reset] = 0.0

    asset: Entity = env.scene[asset_cfg.name]
    x = asset.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
    contact = wheel_contact_any(env)

    # 刚离地：记录起跳位置
    takeoff = state.prev_contact & ~contact
    state.takeoff_x = torch.where(takeoff, x, state.takeoff_x)

    landed = contact & ~state.prev_contact
    advance = x - state.takeoff_x
    in_course = (state.takeoff_x >= float(course_x_min)) & (x <= float(course_x_max))
    ok = landed & (state.air >= int(min_air_steps)) & (advance >= float(min_advance)) & in_course

    state.leaps = state.leaps + ok.long()
    state.air = torch.where(contact, torch.zeros_like(state.air), state.air + 1)
    state.prev_contact = contact.clone()
    state.last_buf = buf.clone()
    state.last_new = ok
    return ok


def leap_bonus(
    env,
    min_air_steps: int = 2,
    min_advance: float = 0.25,
    course_x_min: float = 0.6,
    course_x_max: float = 7.2,
    scale: float = 5.0,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """有效跃沟（离地 >= N 步、且落点前进 >= `min_advance`）→ `+scale`。"""
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    ok = _step(env, min_air_steps, min_advance, course_x_min, course_x_max, asset_cfg)
    return ok.to(torch.float32) * scale


def gaps_crossed(
    env,
    min_air_steps: int = 2,
    min_advance: float = 0.25,
    course_x_min: float = 0.6,
    course_x_max: float = 7.2,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """metric：本回合累计**有效**跃沟次数。"""
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    _step(env, min_air_steps, min_advance, course_x_min, course_x_max, asset_cfg)
    return _state(env).leaps.to(torch.float32)
