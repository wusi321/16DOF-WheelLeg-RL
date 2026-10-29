"""绕桩课程（slalom）的 MDP 项：按顺序经过必经点，碰柱扣分，跳点重罚。

地形（`src/robot/terrains/slalom.py`）在每根柱子旁边放了一个**不参与碰撞的小方块**
作为必经点标记。这里按几何签名把标记读出来、按 x 排序得到顺序，然后：

- `waypoint_reward`    +5   每按顺序经过一个必经点；
- `all_waypoints`      +30  走完全部必经点（一次性，重置时清零）；
- `skip_penalty`       −30  吃到"更后面"的必经点（跳点），**不推进**进度；
- `pillar_hit_penalty` −5   任意轮子贴到柱子（每 0.5 s 最多记一次）；
- `slalom_progress`    metric 已通过 / 总数 —— 直接看这个判断"有没有按顺序走"。

状态按 `episode_length_buf` 缓存，保证同一个 env step 内多个奖励项只更新一次。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .. import naming

WAYPOINT_HALF = 0.02
WAYPOINT_RADIUS = 0.30
PILLAR_MAX_HALF = 0.06
PILLAR_TOUCH_MARGIN = 0.06
SKIP_COOLDOWN_STEPS = 50
HIT_COOLDOWN_STEPS = 25


@dataclass
class _SlalomState:
    waypoints: torch.Tensor | None = None   # (B, K, 2) env 相对坐标
    pillars: torch.Tensor | None = None     # (B, P, 2)
    valid: torch.Tensor | None = None       # (B, K) bool
    idx: torch.Tensor | None = None         # (B,) 下一个必经点序号
    done: torch.Tensor | None = None        # (B,)
    paid: torch.Tensor | None = None        # (B,) all_waypoints 是否已经发过
    skip_cd: torch.Tensor | None = None
    hit_cd: torch.Tensor | None = None
    last_buf: torch.Tensor | None = None
    last_advance: torch.Tensor | None = None
    last_skip: torch.Tensor | None = None
    last_hit: torch.Tensor | None = None


def _wheel_body_ids(robot: Entity) -> list[int]:
    names = [str(n).split("/")[-1] for n in robot.body_names]
    ids: list[int] = []
    for pattern in naming.WHEEL_BODY_PATTERNS:
        leaf = pattern.split("/")[-1]
        if leaf in names:
            ids.append(names.index(leaf))
    return ids


def _model_markers(model) -> list[tuple[float, float]]:
    """必经点标记：不参与碰撞、极薄、截面 = WAYPOINT_HALF。"""
    out = []
    for g in range(model.ngeom):
        if int(model.geom_contype[g]) != 0 or int(model.geom_conaffinity[g]) != 0:
            continue
        size = model.geom_size[g]
        if abs(float(size[2])) > 0.01:
            continue
        if abs(float(size[0]) - WAYPOINT_HALF) > 1e-3:
            continue
        pos = model.geom_pos[g]
        out.append((float(pos[0]), float(pos[1])))
    return out


def _model_pillars(model) -> list[tuple[float, float]]:
    """柱子：参与碰撞、截面小、明显高于地面的方柱。"""
    out = []
    for g in range(model.ngeom):
        if int(model.geom_contype[g]) == 0:
            continue
        size = model.geom_size[g]
        if float(size[0]) > PILLAR_MAX_HALF or float(size[2]) < 0.05:
            continue
        pos = model.geom_pos[g]
        out.append((float(pos[0]), float(pos[1])))
    return out


def _build(env) -> _SlalomState:
    state = _SlalomState()
    model = env.sim._mj_model
    origins = env.scene.env_origins.detach().cpu().numpy()
    n = origins.shape[0]
    marks = _model_markers(model)
    pillars_w = _model_pillars(model)

    per_env: list[list[tuple[float, float]]] = []
    for i in range(n):
        ox, oy = float(origins[i, 0]), float(origins[i, 1])
        pts = sorted(
            (x - ox, y - oy)
            for x, y in marks
            if -0.6 <= x - ox <= 8.4 and abs(y - oy) <= 1.2
        )
        per_env.append(pts)
    k_max = max(1, max(len(p) for p in per_env))
    waypoints = torch.zeros(n, k_max, 2, device=env.device)
    valid = torch.zeros(n, k_max, dtype=torch.bool, device=env.device)
    for i, pts in enumerate(per_env):
        for k, (x, y) in enumerate(pts):
            waypoints[i, k, 0] = x
            waypoints[i, k, 1] = y
            valid[i, k] = True

    per_env_p: list[list[tuple[float, float]]] = []
    for i in range(n):
        ox, oy = float(origins[i, 0]), float(origins[i, 1])
        per_env_p.append(
            [(x - ox, y - oy) for x, y in pillars_w if -0.6 <= x - ox <= 8.4 and abs(y - oy) <= 1.2]
        )
    p_max = max(1, max(len(p) for p in per_env_p))
    pillars = torch.zeros(n, p_max, 2, device=env.device)
    for i, pts in enumerate(per_env_p):
        for k, (x, y) in enumerate(pts):
            pillars[i, k, 0] = x
            pillars[i, k, 1] = y

    state.waypoints = waypoints
    state.valid = valid
    state.pillars = pillars
    state.idx = torch.zeros(n, dtype=torch.long, device=env.device)
    state.done = torch.zeros(n, dtype=torch.bool, device=env.device)
    state.paid = torch.zeros(n, dtype=torch.bool, device=env.device)
    state.skip_cd = torch.zeros(n, dtype=torch.long, device=env.device)
    state.hit_cd = torch.zeros(n, dtype=torch.long, device=env.device)
    state.wheel_ids = _wheel_body_ids(env.scene["robot"])  # type: ignore[attr-defined]
    return state


def _state(env) -> _SlalomState:
    state = getattr(env, "_slalom_state", None)
    if state is None or state.waypoints is None:
        state = _build(env)
        env._slalom_state = state
    return state


def _step(env, asset_cfg: SceneEntityCfg) -> _SlalomState:
    state = _state(env)
    asset: Entity = env.scene[asset_cfg.name]
    buf = env.episode_length_buf.to(torch.long)
    if state.last_buf is not None and torch.equal(buf, state.last_buf):
        return state

    reset = (buf <= 1) if state.last_buf is None else ((buf < state.last_buf) | (buf <= 1))
    if bool(reset.any()):
        state.idx[reset] = 0
        state.done[reset] = False
        state.paid[reset] = False
        state.skip_cd[reset] = 0
        state.hit_cd[reset] = 0

    xy = asset.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
    d = torch.linalg.norm(state.waypoints - xy[:, None, :], dim=2)
    d = torch.where(state.valid, d, torch.full_like(d, 1e3))

    total = state.valid.sum(dim=1).clamp(min=1)
    done = state.idx >= total
    next_idx = state.idx.clamp(max=state.valid.shape[1] - 1)
    d_next = d.gather(1, next_idx[:, None]).squeeze(1)
    reach = (d_next < WAYPOINT_RADIUS) & ~done

    # 跳点：吃到了 idx+1 之后的点（顺序之外）
    ar = torch.arange(state.valid.shape[1], device=env.device)[None, :]
    far = (ar > (state.idx[:, None] + 1)) & state.valid & (d < WAYPOINT_RADIUS)
    skip = far.any(dim=1) & (state.skip_cd == 0)

    advance = reach & ~skip
    state.idx = torch.where(advance, state.idx + 1, state.idx)
    state.done = state.idx >= total

    state.skip_cd = torch.where(
        skip, torch.full_like(state.skip_cd, SKIP_COOLDOWN_STEPS), (state.skip_cd - 1).clamp(min=0)
    )
    state.last_advance = advance
    state.last_skip = skip
    state.last_buf = buf.clone()
    return state


def waypoint_reward(env, scale: float = 5.0, asset_cfg: SceneEntityCfg | None = None) -> torch.Tensor:
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    state = _step(env, asset_cfg)
    return state.last_advance.to(torch.float32) * scale


def skip_penalty(env, scale: float = -30.0, asset_cfg: SceneEntityCfg | None = None) -> torch.Tensor:
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    state = _step(env, asset_cfg)
    return state.last_skip.to(torch.float32) * scale


def pillar_hit_penalty(env, scale: float = -5.0, asset_cfg: SceneEntityCfg | None = None) -> torch.Tensor:
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    state = _step(env, asset_cfg)
    asset: Entity = env.scene[asset_cfg.name]
    ids = state.wheel_ids  # type: ignore[attr-defined]
    pos = asset.data.body_link_pos_w[:, ids] - env.scene.env_origins[:, None, :]
    xy = pos[..., :2]
    diff = (xy[:, :, None, :] - state.pillars[:, None, :, :]).abs()
    hit = (
        (diff[..., 0] <= PILLAR_MAX_HALF + PILLAR_TOUCH_MARGIN)
        & (diff[..., 1] <= PILLAR_MAX_HALF + PILLAR_TOUCH_MARGIN)
    ).any(dim=(1, 2))
    hit = hit & (state.hit_cd == 0)
    state.hit_cd = torch.where(
        hit, torch.full_like(state.hit_cd, HIT_COOLDOWN_STEPS), (state.hit_cd - 1).clamp(min=0)
    )
    return hit.to(torch.float32) * scale


def all_waypoints(env, scale: float = 30.0, asset_cfg: SceneEntityCfg | None = None) -> torch.Tensor:
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    state = _step(env, asset_cfg)
    just = state.done & ~state.paid
    state.paid = state.paid | just
    return just.to(torch.float32) * scale


def slalom_progress(env, asset_cfg: SceneEntityCfg | None = None) -> torch.Tensor:
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    state = _step(env, asset_cfg)
    total = state.valid.sum(dim=1).clamp(min=1)
    return (state.idx.to(torch.float32) / total.to(torch.float32)).clamp(0.0, 1.0)
