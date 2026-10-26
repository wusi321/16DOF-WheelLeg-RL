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

* ``stone_touch_reward``   : ``+w`` for every stone touched for the first time;
* ``all_stones_bonus``     : ``+w`` once when the whole set has been touched;
* ``pit_contact_penalty``  : ``-w`` per step where ``>= min_wheels`` wheels are on
  the pit floor (the "half the feet on the ground" rule);
* ``stone_coverage``       : fraction of stones already touched (metric).

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
XY_TOL = 0.03
#: Allowed mismatch between the wheel's lowest point and the stone top, metres.
Z_TOL = 0.04
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
        grid = np.asarray(patch_origins, dtype=np.float64)  # (num_rows, num_cols, 3)
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


def _touched(env: "ManagerBasedRlEnv", state: _StoneCoverageState) -> torch.Tensor:
    """``(B, S)`` bool: which stones are touched right now."""
    robot = env.scene["robot"]
    if not state.wheel_ids:
        return torch.zeros_like(state.valid)
    # Stones are stored env-relative, so express the wheels in the same frame.
    pos = robot.data.body_link_pos_w[:, state.wheel_ids] - env.scene.env_origins[:, None, :]
    wheel_xy = pos[..., :2]
    bottom = pos[..., 2] - naming.WHEEL_RADIUS

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


def all_stones_bonus(env: "ManagerBasedRlEnv", scale: float = 5.0) -> torch.Tensor:
    """``+scale`` once, when every stone of the patch has been touched."""
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
    scale: float = -1.0,
    min_wheels: int = 2,
    bottom_threshold: float = GROUND_BOTTOM_THRESHOLD,
) -> torch.Tensor:
    """``scale`` per step where at least ``min_wheels`` wheels are on the pit floor."""
    robot = env.scene["robot"]
    state = _state(env)
    if not state.wheel_ids:
        return torch.zeros(env.num_envs, device=env.device)
    pos = robot.data.body_link_pos_w[:, state.wheel_ids]
    bottom = pos[..., 2] - naming.WHEEL_RADIUS - env.scene.env_origins[:, None, 2]
    on_ground = bottom < bottom_threshold
    bad = on_ground.sum(dim=1) >= int(min_wheels)
    return bad.to(torch.float32) * scale


def reset_state(env: "ManagerBasedRlEnv") -> None:
    """Drop the cached state (e.g. after the terrain is rebuilt in play mode)."""
    if hasattr(env, _STATE_ATTR):
        delattr(env, _STATE_ATTR)
