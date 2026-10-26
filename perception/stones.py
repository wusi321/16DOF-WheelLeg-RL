"""Pure-numpy geometry for the "touch every stepping-stone top" rule.

The 2027 rules require the robot to touch **every** stone top at least once while
crossing; this module answers one question without any simulator or ROS dependency:

    given the stone tops (centre, half size, top height) and the positions of the
    wheel contact points, which stones are being touched right now?

The reward code in ``src/robot/mdp/stone_coverage.py`` mirrors the same maths on
torch so it can run on the GPU; this reference implementation is what the unit
tests and the offline evaluation use.

Coordinates: both the stones and the wheels must be expressed in the *same* frame
(world, or env-relative — as long as it is consistent). Heights are metres, with z
pointing up.
"""

from __future__ import annotations

import numpy as np

#: Horizontal tolerance when deciding whether a wheel is over a stone (a wheel is
#: narrower than the stone, so this only forgives discretisation, not skipping).
TOUCH_XY_TOL = 0.03

#: Vertical tolerance between the wheel's lowest point and the stone top.
TOUCH_Z_TOL = 0.02


def touched_stones(
    stone_x: np.ndarray,
    stone_y: np.ndarray,
    stone_half_x: np.ndarray,
    stone_half_y: np.ndarray,
    stone_top: np.ndarray,
    wheel_xy: np.ndarray,
    wheel_bottom_z: np.ndarray,
    xy_tol: float = TOUCH_XY_TOL,
    z_tol: float = TOUCH_Z_TOL,
) -> np.ndarray:
    """Return a boolean mask of the stones currently touched.

    Args:
        stone_x, stone_y: ``(S,)`` stone centres.
        stone_half_x, stone_half_y: ``(S,)`` half extents of the stone top face.
        stone_top: ``(S,)`` z of the stone top.
        wheel_xy: ``(W, 2)`` wheel contact points (wheel centre projected down).
        wheel_bottom_z: ``(W,)`` lowest point of each wheel.
        xy_tol: extra margin around the stone footprint, metres.
        z_tol: allowed mismatch between wheel bottom and stone top, metres.

    Returns:
        ``(S,)`` boolean mask, True where at least one wheel touches the stone.
    """
    sx = np.asarray(stone_x, dtype=np.float64)
    sy = np.asarray(stone_y, dtype=np.float64)
    shx = np.asarray(stone_half_x, dtype=np.float64)
    shy = np.asarray(stone_half_y, dtype=np.float64)
    top = np.asarray(stone_top, dtype=np.float64)
    wxy = np.asarray(wheel_xy, dtype=np.float64).reshape(-1, 2)
    wb = np.asarray(wheel_bottom_z, dtype=np.float64).reshape(-1)

    if not (sx.shape == sy.shape == shx.shape == shy.shape == top.shape):
        raise ValueError("stone arrays must all have the same shape")
    if wxy.shape[0] != wb.shape[0]:
        raise ValueError("wheel_xy and wheel_bottom_z disagree on the wheel count")

    # (S, W)
    dx = np.abs(wxy[None, :, 0] - sx[:, None])
    dy = np.abs(wxy[None, :, 1] - sy[:, None])
    dz = np.abs(wb[None, :] - top[:, None])
    inside = (dx <= shx[:, None] + xy_tol) & (dy <= shy[:, None] + xy_tol)
    height_ok = dz <= z_tol
    return np.any(inside & height_ok, axis=1)


def new_touches(visited: np.ndarray, touched: np.ndarray) -> np.ndarray:
    """Stones touched this step but not before (does not mutate ``visited``)."""
    visited = np.asarray(visited, dtype=bool)
    touched = np.asarray(touched, dtype=bool)
    if visited.shape != touched.shape:
        raise ValueError("visited and touched must have the same shape")
    return touched & ~visited


def pad_stone_table(
    stones: list[tuple[float, float, float, float, float]],
    max_stones: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pack a ragged list of stones into ``(S, 5)`` plus a validity mask.

    Row layout: ``[x, y, half_x, half_y, top]``. Padding rows get half sizes of
    zero so they can never be "touched".
    """
    table = np.zeros((max_stones, 5), dtype=np.float32)
    valid = np.zeros((max_stones,), dtype=bool)
    for i, stone in enumerate(stones[:max_stones]):
        table[i] = np.asarray(stone, dtype=np.float32)
        valid[i] = True
    return table, valid
