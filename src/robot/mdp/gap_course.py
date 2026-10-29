"""沟壑课程（gap course）的 MDP 项：鼓励"越沟"，而不是绕着走或掉下去。

思路：**四轮全部离地 >= N 步之后重新接触地面**，就算一次"成功跃沟"，给一次奖励。
不需要知道沟在哪儿，也不需要和地形参数对齐——只要它真的跳过去了，落地那一刻
就会被记一次。

配套 metric `gaps_crossed` 记录每个回合跳了几次，方便在 wandb 上看"是不是在跳沟"。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

from .stepstone import wheel_contact_any


@dataclass
class _LeapState:
    air: torch.Tensor | None = None            # 截至上一刻连续离地的步数
    leaps: torch.Tensor | None = None          # 本回合成功跃沟次数
    prev_contact: torch.Tensor | None = None   # 上一刻是否有轮子接触
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
    return state


def _step(env, min_air_steps: int) -> torch.Tensor:
    """每个环境步只更新一次，返回"这一步刚刚成功跃沟"的 mask。"""
    state = _state(env)
    buf = env.episode_length_buf.to(torch.long)
    if state.last_buf is not None and torch.equal(buf, state.last_buf):
        return state.last_new  # type: ignore[return-value]

    reset = (buf <= 1) if state.last_buf is None else ((buf < state.last_buf) | (buf <= 1))
    if bool(reset.any()):
        state.air[reset] = 0
        state.leaps[reset] = 0
        state.prev_contact[reset] = True  # 重置这一步不算"落地"

    contact = wheel_contact_any(env)
    landed = contact & ~state.prev_contact
    landed_ok = landed & (state.air >= int(min_air_steps))
    state.leaps = state.leaps + landed_ok.long()
    state.air = torch.where(contact, torch.zeros_like(state.air), state.air + 1)
    state.prev_contact = contact.clone()
    state.last_buf = buf.clone()
    state.last_new = landed_ok
    return landed_ok


def leap_bonus(
    env,
    min_air_steps: int = 2,
    scale: float = 5.0,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """成功跃过一条沟（离地 >= `min_air_steps` 步后落地）→ `+scale`。"""
    del asset_cfg
    return _step(env, min_air_steps).to(torch.float32) * scale


def gaps_crossed(env, min_air_steps: int = 2) -> torch.Tensor:
    """metric：本回合累计跃沟次数。"""
    _step(env, min_air_steps)
    return _state(env).leaps.to(torch.float32)
