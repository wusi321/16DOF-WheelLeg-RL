"""Offline replay of recorded / synthetic terrain frames.

Keeping replay in plain ``.npz`` means phase-2 validation runs on a machine without
ROS 2 and without the robot: the recording side converts a rosbag into these files
(see ``docs/perception/05_roadmap.md``), the training side only reads numpy.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np

from perception.frame import (
    COLS,
    RESOLUTION_M,
    ROWS,
    SCHEMA_VERSION,
    TerrainFrame,
)


def save_frames(path: str | Path, frames: Iterable[TerrainFrame]) -> Path:
    """Write frames to a single ``.npz`` (stacked arrays, one row per frame)."""
    items = list(frames)
    if not items:
        raise ValueError("no frames to save")
    heights = np.stack([f.height_m for f in items]).astype(np.float32)
    valid = np.stack([f.valid for f in items]).astype(bool)
    confidence = np.stack([f.confidence for f in items]).astype(np.float32)
    stamps = np.asarray([f.stamp_ns for f in items], dtype=np.int64)
    statuses = np.asarray([f.status for f in items], dtype="<U16")

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        schema_version=np.int32(SCHEMA_VERSION),
        rows=np.int32(ROWS),
        cols=np.int32(COLS),
        resolution_m=np.float32(RESOLUTION_M),
        stamp_ns=stamps,
        status=statuses,
        height_m=heights,
        valid=valid,
        confidence=confidence,
    )
    return out


def load_frames(path: str | Path) -> list[TerrainFrame]:
    """Read a ``.npz`` written by :func:`save_frames` (or by the rosbag converter)."""
    with np.load(Path(path), allow_pickle=False) as data:
        version = int(data["schema_version"])
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"{path}: schema_version={version}, expected {SCHEMA_VERSION}"
            )
        if int(data["rows"]) != ROWS or int(data["cols"]) != COLS:
            raise ValueError(f"{path}: grid {data['rows']}x{data['cols']} != {ROWS}x{COLS}")
        heights = data["height_m"]
        valid = data["valid"]
        confidence = data["confidence"]
        stamps = data["stamp_ns"]
        statuses = data["status"]
        return [
            TerrainFrame(
                height_m=heights[i],
                valid=valid[i],
                stamp_ns=int(stamps[i]),
                confidence=confidence[i],
                status=str(statuses[i]),
            )
            for i in range(stamps.shape[0])
        ]


class ReplayTerrainSource:
    """A replayable sequence of frames with a ``read()`` API matching the live sources."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.frames = load_frames(self.path)
        self.index = 0

    def __len__(self) -> int:
        return len(self.frames)

    def __iter__(self) -> Iterator[TerrainFrame]:
        for frame in self.frames:
            yield frame.copy()

    def read(self) -> TerrainFrame:
        """Return the next frame, wrapping around at the end (like a looping bag)."""
        frame = self.frames[self.index % len(self.frames)].copy()
        self.index = (self.index + 1) % len(self.frames)
        return frame

    def reset(self) -> None:
        self.index = 0
