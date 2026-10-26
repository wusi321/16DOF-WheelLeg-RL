"""Perception interface for the 16-DOF wheel-leg robot (Odin1 + MuJoCo).

This package is intentionally **free of ROS 2 and mjlab imports**: it only depends on
``numpy`` plus the standard library, so the RL training process never needs a driver
or a sensor connected. See ``docs/perception/`` for the design.

The public surface is:

* :class:`TerrainFrame` -- the single data contract shared by simulation and hardware.
* :func:`encode` -- ``TerrainFrame`` -> policy observation (273 floats).
* ``perception.sources`` -- backends (sim raycast, offline replay).
* ``perception.augment`` -- sim2real randomisation helpers.
"""

from perception.encode import encode, encode_many
from perception.frame import (
    CLIP_MAX,
    CLIP_MIN,
    COLS,
    MISS_SENTINEL,
    OBS_LAYOUT_ID,
    RESOLUTION_M,
    ROWS,
    SCHEMA_VERSION,
    STATUS_ALL_INVALID,
    STATUS_ERROR,
    STATUS_NO_DATA,
    STATUS_OK,
    STATUS_STALE,
    TerrainFrame,
)

__all__ = [
    "CLIP_MAX",
    "CLIP_MIN",
    "COLS",
    "MISS_SENTINEL",
    "OBS_LAYOUT_ID",
    "RESOLUTION_M",
    "ROWS",
    "SCHEMA_VERSION",
    "STATUS_ALL_INVALID",
    "STATUS_ERROR",
    "STATUS_NO_DATA",
    "STATUS_OK",
    "STATUS_STALE",
    "TerrainFrame",
    "encode",
    "encode_many",
]
