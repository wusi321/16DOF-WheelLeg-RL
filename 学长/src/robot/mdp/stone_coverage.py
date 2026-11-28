"""Enforce the 2027 rules' "touch every stepping-stone top" requirement.

The rules require: the robot must leave from the far platform and **every stone top
must be contacted by a foot at least once**; falling off the side, or having half or
more of the feet on the ground (the pit floor), means the obstacle must be redone.

For this wheel-leg robot a "foot contact" is a **wheel touching a stone top**. We do
not need contact forces: a wheel is on a stone when

* its horizontal position is inside the stone's top face (± a few cm), and
* its lowest point (wheel centre minus :data:`naming.WHEEL_RADIUS`) is within a few
  cm of the stone's top height.

Stones are recovered from the compiled scene by name (the terrain builder names them
``stepstone_*``), so no bookkeeping is needed in the terrain generator and the layout
survives curriculum rebuilds.

Rewards provided here (all opt-in, wired up in ``stepstone_env_cfg``):

* ``stone_touch_reward``   : ``+scale`` for every stone touched for the first time;
* ``all_stones_bonus``     : ``+scale`` once when the whole set has been touched;
* ``pit_contact_penalty``  : ``+scale`` per step where ``>= min_wheels`` wheels are on
  the pit floor (the "half the feet on the ground" rule); wire it with a **negative
  weight**;
* ``stone_coverage``       : fraction of stones already touched (metric).

Every function returns an unscaled (non-negative) amount and the reward manager
multiplies it by the configured ``weight``, so ``scale`` defaults to ``1.0`` and the
configured weight *is* the final reward value.

The per-episode state (visited mask) is stored on the environment and cleared when
an episode resets, so no extra manager is required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import mujoco
import numpy as np
import torch

from .. import naming

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

STONE_GEOM_PREFIX = "stepstone_"  # 保留：万一以后地形生成器不再重命名 geom
_STATE_ATTR = "_stepstone_coverage"

#: Extra margin around the stone top face, metres.
#: 5 cm：p4/p5 两轮 coverage 精确停在 0.50-0.52，策略走同侧序列时"路过"另一侧
#: 的桩既不算踩到也没有梯度信号。桩顶只有 0.20 m 宽、轮子高速滚过时停留时间
#: 极短，3 cm 容差会把大量真实踩踏漏记（漏记 = 白踩 = 没有奖励 = 策略学不会）。
XY_TOL = 0.05
#: Allowed mismatch between the wheel's lowest point and the stone top, metres.
#: 5 cm：滚动经过桩顶时的正常颠簸/悬挂行程会让轮底高度瞬时偏离 2-4 cm；
#: 2 cm 容差漏记严重。低难度桩高 3-5 cm 时轮子在平台上不会误判（平台顶与桩顶
#: 齐平、且必须同时满足 XY 在桩面内 + 接触传感器为真，误判需要轮子既在平台
#: 上又在桩顶正上方，物理上不成立）。
Z_TOL = 0.05
#: Wheels below this height (relative to the terrain origin) count as "on the ground".
GROUND_BOTTOM_THRESHOLD = -0.25


@dataclass
class _StoneCoverageState:
    stones: torch.Tensor  # (B, S, 5): x, y, half_x, half_y, top   (env-relative)
    valid: torch.Tensor  # (B, S) bool
    visited: torch.Tensor  # (B, S) bool
    completed: torch.Tensor  # (B,) bool
    wheel_ids: list[int]
    last_len: torch.Tensor | None = None
    last_new: torch.Tensor | None = None
    last_buf: torch.Tensor | None = None
    extras: dict = field(default_factory=dict)


def _geom_name(model: mujoco.MjModel, geom_id: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""


def extract_stones(
    model: mujoco.MjModel,
    half_xy: tuple[float, float] = (0.1, 0.1),
    min_top: float = 0.02,
    size_tol: float = 1e-4,
) -> list[tuple[float, float, float, float, float]]:
    """Return ``[(x, y, half_x, half_y, top), ...]`` for every stone, in world coords.

    The terrain generator renames every geom it creates (``terrain_0``, ``terrain_1``,
    ...), so names cannot be used to identify stones. Instead stones are recognised by
    their geometric signature: a box inside the ``terrain`` body whose horizontal half
    extents equal ``stone_size / 2`` and whose top sits above the platforms. That is
    exactly what :class:`BoxSteppingStonesTerrainCfg` builds, and it is independent of
    per-patch curriculum parameters.
    """
    terrain_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "terrain")
    if terrain_body < 0:
        return []
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    stones: list[tuple[float, float, float, float, float]] = []
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) != terrain_body:
            continue
        if int(model.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_BOX):
            continue
        size = model.geom_size[g]
        if abs(float(size[0]) - half_xy[0]) > size_tol or abs(float(size[1]) - half_xy[1]) > size_tol:
            continue
        pos = data.geom_xpos[g]
        top = float(pos[2] + size[2])
        if top < min_top:
            continue
        stones.append((float(pos[0]), float(pos[1]), float(size[0]), float(size[1]), top))
    return stones


def _wheel_body_ids(robot) -> list[int]:
    """Indices of the four wheel bodies (scene body names carry an entity prefix)."""
    names = [str(n).split("/")[-1] for n in robot.body_names]
    ids: list[int] = []
    for pattern in naming.WHEEL_BODY_PATTERNS:
        leaf = pattern.split("/")[-1]
        if leaf in names:
            ids.append(names.index(leaf))
    return ids


def _stepstone_cfg(env: "ManagerBasedRlEnv"):
    """Return the ``BoxSteppingStonesTerrainCfg`` instance used by this env, if any."""
    terrain = getattr(env.cfg.scene, "terrain", None)
    generator = getattr(terrain, "terrain_generator", None)
    for cfg in (getattr(generator, "sub_terrains", None) or {}).values():
        if hasattr(cfg, "stone_size") and hasattr(cfg, "start_length"):
            return cfg
    return None


def _build_state(env: "ManagerBasedRlEnv") -> _StoneCoverageState:
    device = env.device
    model = env.sim._mj_model
    stone_cfg = _stepstone_cfg(env)
    if stone_cfg is not None:
        half_xy = (stone_cfg.stone_size[0] / 2.0, stone_cfg.stone_size[1] / 2.0)
        # Patch centre relative to the spawn origin (spawn is at the start platform centre).
        center_rel = (stone_cfg.size[0] / 2.0 - stone_cfg.start_length / 2.0, 0.0)
    else:
        half_xy = (0.1, 0.1)
        center_rel = (1.6, 0.0)

    stones_world = extract_stones(model, half_xy=half_xy)
    origins = env.scene.env_origins.detach().cpu().numpy()
    num_envs = origins.shape[0]

    # 按**地形 patch**分组，而不是按 env 出生点就近分配：所有 env 在初始课程下都
    # 出生在 level 0 的同一个 patch 上（`max_init_terrain_level=0`），按“最近的
    # env 出生点”分配会把整片地形的桩全塞给 env 0，其余 env 一根桩都没有
    # （于是 `all_stones` 对它们白送奖励）。正确做法是用 patch 网格索引。
    terrain = env.scene.terrain
    patch_origins = getattr(terrain, "terrain_origins", None)
    levels = getattr(terrain, "terrain_levels", None)
    types = getattr(terrain, "terrain_types", None)

    per_env: list[list[tuple[float, float, float, float, float]]] = [[] for _ in range(num_envs)]

    if patch_origins is not None and levels is not None and types is not None:
        # ``terrain_origins`` lives on the env device (CUDA when training), so it must be
        # moved to host before numpy can touch it.
        patch_origins_host = patch_origins
        if hasattr(patch_origins_host, "detach"):
            patch_origins_host = patch_origins_host.detach().cpu().numpy()
        grid = np.asarray(patch_origins_host, dtype=np.float64)  # (num_rows, num_cols, 3)
        rows, cols = grid.shape[0], grid.shape[1]
        # Patches sit 4.0 m apart along x and are ~4.0 m long, so grouping by "nearest
        # patch centre" leaks stones into the neighbouring row (a stone at dx=2.6 is
        # closer to the next patch's centre). Assign by rectangle containment instead.
        size_x = float(stone_cfg.size[0]) if stone_cfg is not None else 4.0
        size_y = float(stone_cfg.size[1]) if stone_cfg is not None else 1.6
        patch_stones: dict[tuple[int, int], list[tuple[float, float, float, float, float]]] = {}
        for x, y, hx, hy, top in stones_world:
            best: tuple[int, int] | None = None
            best_d = float("inf")
            for r in range(rows):
                for c in range(cols):
                    ox, oy, oz = (float(v) for v in grid[r, c])
                    dx, dy = x - ox, y - oy
                    if -0.5 <= dx <= size_x + 0.5 and abs(dy) <= size_y / 2.0 + 0.5:
                        d = abs(dx - center_rel[0]) + abs(dy)
                        if d < best_d:
                            best_d, best = d, (r, c)
            if best is None:  # 兜底：最近的 patch 中心
                d2 = (grid[:, :, 0] - x) ** 2 + (grid[:, :, 1] - y) ** 2
                r, c = np.unravel_index(int(np.argmin(d2)), (rows, cols))
                best = (int(r), int(c))
            r, c = best
            ox, oy, oz = (float(v) for v in grid[r, c])
            patch_stones.setdefault((r, c), []).append((x - ox, y - oy, hx, hy, top - oz))

        levels_np = levels.detach().cpu().numpy()
        types_np = types.detach().cpu().numpy()
        for i in range(num_envs):
            key = (int(levels_np[i]) % rows, int(types_np[i]) % cols)
            for dx, dy, hx, hy, dz in patch_stones.get(key, []):
                per_env[i].append(
                    (dx + float(grid[key[0], key[1], 0] - origins[i, 0]),
                     dy + float(grid[key[0], key[1], 1] - origins[i, 1]),
                     hx, hy,
                     dz + float(grid[key[0], key[1], 2] - origins[i, 2]))
                )
    else:  # 非生成式地形：退回“最近的出生点”启发式
        centers_x = origins[:, 0] + center_rel[0]
        centers_y = origins[:, 1] + center_rel[1]
        for x, y, hx, hy, top in stones_world:
            d2 = (centers_x - x) ** 2 + (centers_y - y) ** 2
            best = int(np.argmin(d2))
            per_env[best].append(
                (x - origins[best, 0], y - origins[best, 1], hx, hy, top - origins[best, 2])
            )

    max_stones = max(1, max(len(s) for s in per_env))
    stones = torch.zeros((num_envs, max_stones, 5), dtype=torch.float32, device=device)
    valid = torch.zeros((num_envs, max_stones), dtype=torch.bool, device=device)
    for i, env_stones in enumerate(per_env):
        if not env_stones:
            continue
        arr = torch.as_tensor(env_stones, dtype=torch.float32, device=device)
        stones[i, : arr.shape[0]] = arr
        valid[i, : arr.shape[0]] = True

    robot = env.scene["robot"]
    state = _StoneCoverageState(
        stones=stones,
        valid=valid,
        visited=torch.zeros_like(valid),
        completed=torch.zeros(num_envs, dtype=torch.bool, device=device),
        wheel_ids=_wheel_body_ids(robot),
    )
    state.extras["counts"] = valid.sum(dim=1)
    return state


def _state(env: "ManagerBasedRlEnv") -> _StoneCoverageState:
    state = getattr(env, _STATE_ATTR, None)
    if state is None:
        state = _build_state(env)
        setattr(env, _STATE_ATTR, state)
    return state


def _wheel_contact(env: "ManagerBasedRlEnv", num_wheels: int) -> torch.Tensor | None:
    """``(B, W)`` bool of wheel-ground contacts, or ``None`` when no sensor is present."""
    try:
        sensor = env.scene["feet_ground_contact"]
    except (KeyError, ValueError):
        return None
    found = sensor.data.found
    if found.ndim != 2 or found.shape[1] != num_wheels:
        return None
    return found > 0.5


def _touched(env: "ManagerBasedRlEnv", state: _StoneCoverageState) -> torch.Tensor:
    """``(B, S)`` bool: which stones are touched right now.

    A wheel counts as touching a stone only when it is *in contact* (contact sensor
    ``found``) as well as geometrically on top of it. Without the contact check a
    fast robot that flies over a stone gets credited for "touching" it.
    """
    robot = env.scene["robot"]
    if not state.wheel_ids:
        return torch.zeros_like(state.valid)
    # Stones are stored env-relative, so express the wheels in the same frame.
    pos = robot.data.body_link_pos_w[:, state.wheel_ids] - env.scene.env_origins[:, None, :]
    wheel_xy = pos[..., :2]
    bottom = pos[..., 2] - naming.WHEEL_RADIUS
    contact = _wheel_contact(env, len(state.wheel_ids))

    sx = state.stones[:, None, :, 0]  # (B, 1, S)
    sy = state.stones[:, None, :, 1]
    shx = state.stones[:, None, :, 2]
    shy = state.stones[:, None, :, 3]
    top = state.stones[:, None, :, 4]

    dx = (wheel_xy[:, :, None, 0] - sx).abs()  # (B, W, S)
    dy = (wheel_xy[:, :, None, 1] - sy).abs()
    dz = (bottom[:, :, None] - top).abs()
    inside = (dx <= shx + XY_TOL) & (dy <= shy + XY_TOL)
    hit = inside & (dz <= Z_TOL) & state.valid[:, None, :]
    if contact is not None:
        hit = hit & contact[:, :, None]
    return hit.any(dim=1)


def _sync(env: "ManagerBasedRlEnv", state: _StoneCoverageState) -> torch.Tensor:
    """Update the visited mask once per env step and return newly touched stones.

    The per-step cache is keyed on a **clone** of ``episode_length_buf``: the
    environment increments that buffer in place, so storing the tensor itself would
    make every later comparison trivially equal and freeze the cache forever.
    """
    buf = env.episode_length_buf.to(torch.long)
    if state.last_buf is not None and torch.equal(buf, state.last_buf):
        return state.last_new

    if state.last_len is None:
        reset = buf <= 1
    else:
        reset = (buf < state.last_len) | (buf <= 1)
    if reset.any():
        state.visited[reset] = False
        state.completed[reset] = False

    touched = _touched(env, state)
    newly = touched & ~state.visited
    state.visited |= touched
    state.last_len = buf.clone()
    state.last_new = newly
    state.last_buf = buf.clone()
    return newly


def stone_touch_reward(env: "ManagerBasedRlEnv", scale: float = 1.0) -> torch.Tensor:
    """``+scale`` for each stone touched for the first time in this episode."""
    state = _state(env)
    newly = _sync(env, state)
    return newly.sum(dim=1).to(torch.float32) * scale


def all_stones_bonus(env: "ManagerBasedRlEnv", scale: float = 1.0) -> torch.Tensor:
    """``+scale`` once, when every stone of the patch has been touched.

    ``scale`` defaults to 1.0 because the reward manager already multiplies the
    returned amount by the configured weight (``weight=20`` -> ``+20``).  A non-1.0
    default here silently multiplied the configured bonus by 5.
    """
    state = _state(env)
    _sync(env, state)
    done = state.visited | ~state.valid
    # An env whose patch has no stones must not collect the bonus for free.
    all_done = done.all(dim=1) & (state.valid.sum(dim=1) > 0)
    just = all_done & ~state.completed
    state.completed = state.completed | just
    return just.to(torch.float32) * scale


def stone_coverage(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Fraction of stones touched so far (metric, [0, 1])."""
    state = _state(env)
    _sync(env, state)
    total = state.valid.sum(dim=1).clamp(min=1)
    return (state.visited & state.valid).sum(dim=1).to(torch.float32) / total


def pit_contact_penalty(
    env: "ManagerBasedRlEnv",
    scale: float = 1.0,
    min_wheels: int = 2,
    bottom_threshold: float = GROUND_BOTTOM_THRESHOLD,
) -> torch.Tensor:
    """``scale`` per step where at least ``min_wheels`` wheels are on the pit floor.

    Returns a non-negative amount; the configured **negative** weight turns it into a
    penalty.  ``scale`` used to default to ``-1.0`` while the config also used
    ``weight=-2.0``, so the two negatives cancelled into a ``+2``/step *reward* for
    falling into the pit.
    """
    robot = env.scene["robot"]
    state = _state(env)
    if not state.wheel_ids:
        return torch.zeros(env.num_envs, device=env.device)
    pos = robot.data.body_link_pos_w[:, state.wheel_ids]
    bottom = pos[..., 2] - naming.WHEEL_RADIUS - env.scene.env_origins[:, None, 2]
    on_ground = bottom < bottom_threshold
    bad = on_ground.sum(dim=1) >= int(min_wheels)
    return bad.to(torch.float32) * scale


def landing_precision_reward(
    env: "ManagerBasedRlEnv",
    scale: float = 1.0,
    x_min: float = 0.8,
    x_max: float = 6.8,
    bottom_tol: float = 0.06,
    min_speed: float = 0.25,
) -> torch.Tensor:
    """Per-wheel reward for landing *near the centre* of a stone top (0..1 each).

    A wheel that sits on a stone top rectangle scores ``(1 - normalized_distance)^2``
    where the distance is measured from the stone centre in units of the stone's
    half-extent along each axis.  Wheels on the very edge score ~0, wheels off the
    stones score 0.  This pins the foot placement to the middle of the tops instead
    of the front/outer edge that the coarse "inside the rectangle" check tolerated.

    The reward is gated on the robot actually moving (body horizontal speed above
    ``min_speed``): otherwise a policy discovered that simply standing still on a
    stone farms this static term indefinitely (episode lengths blew up to the
    30 s timeout with coverage high but the goal never reached).
    """
    state = _state(env)
    robot = env.scene["robot"]
    if not state.wheel_ids or state.stones.numel() == 0:
        return torch.zeros(env.num_envs, device=env.device)
    origins = env.scene.env_origins
    pos = robot.data.body_link_pos_w[:, state.wheel_ids] - origins[:, None, :]
    wheel_xy = pos[..., :2]
    bottom = pos[..., 2] - naming.WHEEL_RADIUS
    stones = state.stones
    valid = state.valid
    sx, sy = stones[..., 0], stones[..., 1]
    shx = stones[..., 2].clamp(min=1e-3)
    shy = stones[..., 3].clamp(min=1e-3)
    top = stones[..., 4]
    dx = (wheel_xy[:, :, None, 0] - sx[:, None, :]).abs()
    dy = (wheel_xy[:, :, None, 1] - sy[:, None, :]).abs()
    inside = (dx <= shx[:, None, :]) & (dy <= shy[:, None, :]) & valid[:, None, :]
    on_top = inside & ((bottom[:, :, None] - top[:, None, :]).abs() <= bottom_tol)
    norm = (dx / shx[:, None, :]) ** 2 + (dy / shy[:, None, :]) ** 2
    norm = norm.clamp(max=1.0).sqrt()
    # 严格版（用户 2026-10-30 回放反馈“实际情况下已经掉下去了”）：轮心必须
    # 落在桩心附近——归一化距离 <= 0.5（约 5 cm）才按线性给分，超出 5 cm 直接 0。
    prec = ((0.5 - norm).clamp(min=0.0) / 0.5) * on_top.to(torch.float32)
    best = prec.max(dim=-1).values  # (N, W)
    in_field = (pos[..., 0] >= x_min) & (pos[..., 0] <= x_max)
    speed = robot.data.body_link_lin_vel_w[:, 0, :2].norm(dim=-1)
    moving = speed >= min_speed
    return (
        (best * in_field.to(torch.float32)).sum(dim=1)
        * moving.to(torch.float32)
        * scale
    )


def landing_edge_penalty(
    env: "ManagerBasedRlEnv",
    scale: float = 1.0,
    x_min: float = 0.8,
    x_max: float = 6.8,
    bottom_tol: float = 0.06,
    edge_frac: float = 0.5,
    min_speed: float = 0.25,
) -> torch.Tensor:
    """每步按“超出桩心允许范围的程度”累计负分（0–4 轮）。

    返回非负量，由奖励管理器配负 weight 变成惩罚。``edge_frac=0.5`` 表示轮心的
    归一化中心距离超过 0.5（约 5 cm）就开始线性计罚，越贴边罚得越重（到桩边
    归一化 1.0 时每个轮子计满 1.0）。用户 2026-10-30 回放指出“轮子都已经到桩边
    了，实机已经掉下去了”，所以这项和 ``landing_precision_reward`` 一起把
    合格落点从“整个桩顶面”收紧到“桩心附近 5 cm”。按机身移动做门控，防站桩刷分。
    """
    state = _state(env)
    robot = env.scene["robot"]
    if not state.wheel_ids or state.stones.numel() == 0:
        return torch.zeros(env.num_envs, device=env.device)
    origins = env.scene.env_origins
    pos = robot.data.body_link_pos_w[:, state.wheel_ids] - origins[:, None, :]
    wheel_xy = pos[..., :2]
    bottom = pos[..., 2] - naming.WHEEL_RADIUS
    stones = state.stones
    valid = state.valid
    sx, sy = stones[..., 0], stones[..., 1]
    shx = stones[..., 2].clamp(min=1e-3)
    shy = stones[..., 3].clamp(min=1e-3)
    top = stones[..., 4]
    dx = (wheel_xy[:, :, None, 0] - sx[:, None, :]).abs()
    dy = (wheel_xy[:, :, None, 1] - sy[:, None, :]).abs()
    inside = (dx <= shx[:, None, :]) & (dy <= shy[:, None, :]) & valid[:, None, :]
    on_top = inside & ((bottom[:, :, None] - top[:, None, :]).abs() <= bottom_tol)
    norm = ((dx / shx[:, None, :]) ** 2 + (dy / shy[:, None, :]) ** 2)
    norm = norm.clamp(max=1.0).sqrt()
    in_field = (pos[..., 0] >= x_min) & (pos[..., 0] <= x_max)
    span = max(1e-3, 1.0 - float(edge_frac))
    over = ((norm - float(edge_frac)) / span).clamp(min=0.0)
    # 每个轮子取它所在桩的最小超出量（轮子只对“它踩的那根桩”负责）
    over = (over * on_top.to(torch.float32)).max(dim=-1).values
    over = over * in_field.to(torch.float32)
    value = over.sum(dim=1)
    speed = robot.data.body_link_lin_vel_w[:, 0, :2].norm(dim=-1)
    moving = speed >= min_speed
    return value * moving.to(torch.float32) * scale


def rear_gap_clearance_penalty(
    env: "ManagerBasedRlEnv",
    scale: float = 1.0,
    margin: float = 0.06,
    x_min: float = 0.8,
    x_max: float = 6.8,
    y_limit: float = 0.9,
) -> torch.Tensor:
    """Penalty for rear wheels dipping below the stone tops while crossing gaps.

    For each rear wheel that is over the stone field but **outside** every stone top
    rectangle (i.e. it is over the gap between two stones), this measures how far the
    wheel centre has dropped below the nearest stone top relative to ``margin``::

        shortfall = max(0, margin - (wheel_center_z - nearest_stone_top_z))

    The shortfalls of the two rear wheels are summed.  Returns a non-negative amount;
    the configured **negative** weight turns it into a penalty.  This directly punishes
    the "front legs leap, rear wheels sag into the gap" gait, which simulates fine on
    soft contacts but would snag on the real stone edges.
    """
    state = _state(env)
    robot = env.scene["robot"]
    if not state.wheel_ids or state.stones.numel() == 0:
        return torch.zeros(env.num_envs, device=env.device)
    origins = env.scene.env_origins
    pos = robot.data.body_link_pos_w[:, state.wheel_ids] - origins[:, None, :]  # (N,4,3)
    rear = pos[:, -2:]  # FL, FR, RL, RR -> rear = RL, RR
    stones = state.stones  # (N,S,5) = x, y, half_x, half_y, top (relative to env origin)
    valid = state.valid  # (N,S)
    sxy = stones[..., :2]
    shx = stones[..., 2]
    shy = stones[..., 3]
    sz = stones[..., 4]
    big = torch.full_like(sxy[..., 0], float("inf"))

    penalty = torch.zeros(env.num_envs, device=env.device)
    for w in range(rear.shape[1]):
        px = rear[:, w, 0]
        py = rear[:, w, 1]
        pz = rear[:, w, 2]
        in_field = (px >= x_min) & (px <= x_max) & (py.abs() <= y_limit)
        inside = (
            (px[:, None] - sxy[..., 0]).abs() <= shx
        ) & (
            (py[:, None] - sxy[..., 1]).abs() <= shy
        ) & valid
        over_gap = in_field & ~inside.any(dim=1)
        d2 = (sxy[..., 0] - px[:, None]) ** 2 + (sxy[..., 1] - py[:, None]) ** 2
        d2 = torch.where(valid, d2, big)
        j = d2.argmin(dim=1)
        near_top = sz.gather(1, j[:, None]).squeeze(1)
        shortfall = torch.clamp(margin - (pz - near_top), min=0.0)
        penalty = penalty + torch.where(over_gap, shortfall, torch.zeros_like(shortfall))
    return penalty * scale


def reset_state(env: "ManagerBasedRlEnv") -> None:
    """Drop the cached state (e.g. after the terrain is rebuilt in play mode)."""
    if hasattr(env, _STATE_ATTR):
        delattr(env, _STATE_ATTR)
