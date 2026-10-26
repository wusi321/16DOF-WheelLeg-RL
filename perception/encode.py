"""``TerrainFrame`` -> policy observation.

The encoding rule is frozen (see ``docs/perception/04_local_terrain_obs.md``):

    obs = clip(valid ? height_m : MISS_SENTINEL, CLIP_MIN, CLIP_MAX)

which reproduces, element by element, what the existing MuJoCo observation pipeline
produces (``height_scan`` fills misses with the sensor's ``max_distance`` and the
observation term clips to ``(-1, 1)``). Any change here must bump
:data:`perception.OBS_LAYOUT_ID` and use a new task id.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from perception.frame import (
    CLIP_MAX,
    CLIP_MIN,
    MISS_SENTINEL,
    TerrainFrame,
    flat_index,
)

OBS_DIM = 273


def encode(frame: TerrainFrame) -> np.ndarray:
    """Encode one frame into a ``float32[273]`` observation vector."""
    if frame.schema_version != 1:
        raise ValueError(f"unsupported schema_version {frame.schema_version}")
    h = np.where(frame.valid, frame.height_m, np.float32(MISS_SENTINEL))
    out = np.clip(h, CLIP_MIN, CLIP_MAX).astype(np.float32, copy=False)
    return np.ascontiguousarray(out.reshape(-1))


def encode_many(frames: Iterable[TerrainFrame]) -> np.ndarray:
    """Encode several frames into ``float32[N, 273]``."""
    rows: list[np.ndarray] = [encode(f) for f in frames]
    if not rows:
        return np.zeros((0, OBS_DIM), dtype=np.float32)
    return np.stack(rows, axis=0)


def decode(obs: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Split a flat observation back into ``(height_m, valid)`` for debugging.

    ``valid`` is recovered from the sentinel: a cell encoded as exactly
    ``CLIP_MAX`` is reported as invalid. This is lossy for genuinely clipped
    cells deeper than 1 m, which is fine for visualisation only.
    """
    arr = np.asarray(obs, dtype=np.float32).reshape(-1)
    if arr.size != OBS_DIM:
        raise ValueError(f"expected {OBS_DIM} values, got {arr.size}")
    grid = arr.reshape(-1, 21)
    valid = grid < CLIP_MAX
    return grid.astype(np.float32), valid


def encode_cell(values: dict[tuple[int, int], float], default: float = 0.32) -> np.ndarray:
    """Convenience for tests: build an observation from a few ``(row, col) -> height`` cells."""
    grid = np.full((273,), default, dtype=np.float32)
    for (row, col), value in values.items():
        grid[flat_index(row, col)] = value
    return np.clip(grid, CLIP_MIN, CLIP_MAX)
