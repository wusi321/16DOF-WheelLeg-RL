"""Sim2real randomisation for training (never used on hardware).

The defaults follow ``docs/perception/03_sim2real_mapping.md``:

* range noise ``sigma(d) = 0.01 + 0.004 * d`` (0.5 m -> 1.2 cm, 5 m -> 3 cm);
* dropout 5 % below 3 m, 15 % between 3 and 6 m, 40 % beyond 6 m;
* a static FOV mask can hide cells the real sensor cannot see;
* :class:`DelayedSource` emulates cloud latency / 10 Hz updates.

Distance is approximated by the height itself (downward rays): a cell at height
``h`` is roughly ``h`` metres away from the base.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from perception.frame import STATUS_STALE, TerrainFrame


class Source(Protocol):
    def read(self) -> TerrainFrame: ...


def add_range_noise(
    frame: TerrainFrame,
    rng: np.random.Generator,
    sigma_at_zero: float = 0.01,
    slope_per_m: float = 0.004,
) -> TerrainFrame:
    """Add distance-dependent Gaussian noise on valid cells (metres)."""
    out = frame.copy()
    sigma = sigma_at_zero + slope_per_m * np.abs(frame.height_m)
    noise = rng.normal(0.0, sigma).astype(np.float32)
    out.height_m = np.where(frame.valid, frame.height_m + noise, frame.height_m)
    return out


def random_dropout(
    frame: TerrainFrame,
    rng: np.random.Generator,
    bands: Sequence[tuple[float, float]] = ((3.0, 0.05), (6.0, 0.15), (np.inf, 0.40)),
) -> TerrainFrame:
    """Mark a distance-dependent share of cells as invalid to mimic sparsity.

    ``bands`` must be sorted by ascending upper bound; the last entry should use
    ``np.inf`` as its upper bound.
    """
    out = frame.copy()
    h = np.abs(frame.height_m)
    prob = np.full(h.shape, float(bands[-1][1]), dtype=np.float32)
    for upper, p in reversed(bands[:-1]):
        prob = np.where(h <= upper, np.float32(p), prob)
    drop = rng.random(h.shape) < prob
    out.valid = frame.valid & ~drop
    if not out.valid.any():
        out.status = "all_invalid"
    return out


def apply_static_mask(frame: TerrainFrame, mask: np.ndarray) -> TerrainFrame:
    """Hide cells where ``mask`` is False (mounting-dependent blind spots)."""
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != frame.valid.shape:
        raise ValueError(f"mask shape {mask.shape} != {frame.valid.shape}")
    out = frame.copy()
    out.valid = frame.valid & mask
    return out


def make_rect_mask(
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> np.ndarray:
    """Boolean mask of the grid cells inside the given base-frame rectangle."""
    from perception.frame import COLS, RESOLUTION_M, ROWS, X_MIN, Y_MIN

    xs = X_MIN + RESOLUTION_M * np.arange(COLS)
    ys = Y_MIN + RESOLUTION_M * np.arange(ROWS)
    in_x = (xs >= x_range[0]) & (xs <= x_range[1])
    in_y = (ys >= y_range[0]) & (ys <= y_range[1])
    return np.outer(in_y, in_x)


def smooth(frame: TerrainFrame, kernel: int = 3) -> TerrainFrame:
    """Optional 3x3 median filter over valid cells (reduces single-point spikes)."""
    if kernel % 2 == 0 or kernel < 3:
        raise ValueError("kernel must be odd and >= 3")
    out = frame.copy()
    pad = kernel // 2
    h = np.pad(frame.height_m, pad, mode="edge")
    v = np.pad(frame.valid.astype(np.uint8), pad, mode="edge").astype(bool)
    acc = np.zeros_like(frame.height_m)
    cnt = np.zeros_like(frame.height_m)
    for dy in range(kernel):
        for dx in range(kernel):
            acc += np.where(v[dy : dy + frame.height_m.shape[0], dx : dx + frame.height_m.shape[1]],
                            h[dy : dy + frame.height_m.shape[0], dx : dx + frame.height_m.shape[1]], 0.0)
            cnt += v[dy : dy + frame.height_m.shape[0], dx : dx + frame.height_m.shape[1]]
    with np.errstate(invalid="ignore", divide="ignore"):
        filled = np.where(cnt > 0, acc / np.maximum(cnt, 1), frame.height_m)
    out.height_m = np.where(frame.valid, filled, frame.height_m).astype(np.float32)
    return out


class DelayedSource:
    """Wrap a source and return a frame that is ``delay_steps`` frames old.

    Emulates the real pipeline: 10 Hz clouds consumed by a 50 Hz policy, plus
    transport/compute latency. When the buffer is empty the newest frame is
    returned with ``status="stale"`` so callers can degrade explicitly.
    """

    def __init__(self, source: Source, delay_steps: int = 1) -> None:
        if delay_steps < 0:
            raise ValueError("delay_steps must be >= 0")
        self.source = source
        self.delay_steps = int(delay_steps)
        self._buffer: deque[TerrainFrame] = deque(maxlen=delay_steps + 1)

    def read(self) -> TerrainFrame:
        frame = self.source.read()
        self._buffer.append(frame)
        if len(self._buffer) <= self.delay_steps:
            out = self._buffer[0].copy()
            out.status = STATUS_STALE
            return out
        return self._buffer[0].copy()
