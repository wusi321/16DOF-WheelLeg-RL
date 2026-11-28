"""MuJoCo height-scanner -> :class:`TerrainFrame`.

The class is intentionally duck-typed: it only needs an object exposing
``data.distances``, ``data.frame_pos_w`` and ``data.hit_pos_w`` (a mjlab
``RayCastSensor`` does), so the training process never imports mjlab here and the
class is trivially testable with plain arrays via :meth:`from_arrays`.

Semantics (must match ``mjlab.envs.mdp.observations.height_scan``):

    height = frame_z - hit_z          # positive: ground below the base
    valid  = distance >= 0            # negative distance means "no hit"

Rays are ordered frame-major, x contiguous; the grid is reshaped to (13, 21).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from perception.frame import (
    COLS,
    MISS_SENTINEL,
    ROWS,
    STATUS_ALL_INVALID,
    STATUS_OK,
    TerrainFrame,
)


def _to_numpy(value: Any) -> np.ndarray:
    """Convert torch / warp arrays to numpy without importing torch."""
    if isinstance(value, np.ndarray):
        return value
    for attr in ("detach",):
        if hasattr(value, attr):
            value = getattr(value, attr)()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


class SimTerrainSource:
    """Read one frame per call from a mjlab raycast sensor (one env per source)."""

    def __init__(self, sensor: Any, env_index: int = 0, stamp_ns: int | None = None) -> None:
        self.sensor = sensor
        self.env_index = int(env_index)
        self.stamp_ns = stamp_ns

    def read(self) -> TerrainFrame:
        data = self.sensor.data
        distances = _to_numpy(data.distances)[self.env_index]  # (N,)
        frame_pos = _to_numpy(data.frame_pos_w)[self.env_index]  # (F, 3)
        hit_pos = _to_numpy(data.hit_pos_w)[self.env_index]  # (F, N, 3)

        frame_z = np.asarray(frame_pos, dtype=np.float32)[..., 2]  # (F,)
        hit_z = np.asarray(hit_pos, dtype=np.float32)[..., 2]  # (F, N)
        if hit_z.ndim == 1:
            hit_z = hit_z[None, :]
        heights = frame_z[:, None] - hit_z  # (F, N)

        return self.from_arrays(
            np.asarray(distances, dtype=np.float32).reshape(-1),
            heights.reshape(-1),
            stamp_ns=self.stamp_ns if self.stamp_ns is not None else 0,
        )

    @staticmethod
    def from_arrays(
        distances: np.ndarray,
        heights: np.ndarray,
        stamp_ns: int = 0,
    ) -> TerrainFrame:
        """Build a frame from per-ray ``distances`` and already-computed ``heights``."""
        dist = np.asarray(distances, dtype=np.float32).reshape(-1)
        h = np.asarray(heights, dtype=np.float32).reshape(-1)
        if dist.size != h.size:
            raise ValueError(f"distances {dist.size} != heights {h.size}")
        if dist.size != ROWS * COLS:
            raise ValueError(f"expected {ROWS * COLS} rays, got {dist.size}")

        valid = dist >= 0.0
        heights = np.where(valid, h, np.float32(MISS_SENTINEL))
        grid = heights.reshape(ROWS, COLS)
        valid_grid = valid.reshape(ROWS, COLS)
        status = STATUS_OK if valid_grid.any() else STATUS_ALL_INVALID
        return TerrainFrame(
            height_m=grid.astype(np.float32),
            valid=valid_grid,
            stamp_ns=stamp_ns,
            status=status,
        )
