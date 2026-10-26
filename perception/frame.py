"""The single terrain-observation contract shared by simulation and hardware.

All geometry constants mirror the MuJoCo height scanner configured in
``src/robot/config/env_cfgs.py`` (``GridPatternCfg(size=(1.6, 1.0), resolution=0.08)``)
and verified against the running environment:

* 13 rows (y) x 21 columns (x), flattened with x contiguous (``index = row * 21 + col``);
* x in [-0.80, +0.80] m (symmetric), y in [-0.50, +0.46] m (asymmetric, 2 cm bias to the left);
* rays point straight down in a yaw-aligned ``base_link`` frame;
* ``height = z_base - z_ground`` (positive means the ground is *below* the base,
  so higher obstacles have *smaller* heights).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: Bumped whenever the on-disk or in-memory layout changes in a breaking way.
SCHEMA_VERSION = 1

#: Identifier recorded next to checkpoints so a layout change cannot silently
#: be mixed with an old policy.
OBS_LAYOUT_ID = "hscan13x21x0.08+yaw+clip1+v1"

ROWS = 13
COLS = 21
RESOLUTION_M = 0.08
X_MIN, X_MAX = -0.80, 0.80
Y_MIN, Y_MAX = -0.50, 0.46

#: Value a missed ray is encoded with *before* clipping, matching
#: ``height_scan(..., miss_value=sensor.cfg.max_distance)`` with ``max_distance=5.0``.
MISS_SENTINEL = 5.0
CLIP_MIN, CLIP_MAX = -1.0, 1.0

STATUS_OK = "ok"
STATUS_STALE = "stale"
STATUS_NO_DATA = "no_data"
STATUS_ALL_INVALID = "all_invalid"
STATUS_ERROR = "error"

_STATUSES = {STATUS_OK, STATUS_STALE, STATUS_NO_DATA, STATUS_ALL_INVALID, STATUS_ERROR}


class TerrainFrameError(ValueError):
    """Raised when a frame does not satisfy the contract."""


def grid_centers() -> tuple[np.ndarray, np.ndarray]:
    """Return the (x, y) coordinates of every cell centre, shape ``(ROWS, COLS)``."""
    xs = X_MIN + RESOLUTION_M * np.arange(COLS, dtype=np.float32)
    ys = Y_MIN + RESOLUTION_M * np.arange(ROWS, dtype=np.float32)
    # Rows are y (slow), columns are x (fast) so that ravel() is x-contiguous.
    return np.meshgrid(xs, ys)


def cell_of(x: float, y: float) -> tuple[int, int]:
    """Map a point in the base frame to ``(row, col)``; out-of-grid raises ``IndexError``."""
    col = int(round((x - X_MIN) / RESOLUTION_M))
    row = int(round((y - Y_MIN) / RESOLUTION_M))
    if not (0 <= row < ROWS and 0 <= col < COLS):
        raise IndexError(f"point ({x:.3f}, {y:.3f}) is outside the {ROWS}x{COLS} grid")
    return row, col


def flat_index(row: int, col: int) -> int:
    """Index into the flat 273-vector for ``(row, col)``."""
    if not (0 <= row < ROWS and 0 <= col < COLS):
        raise IndexError(f"cell ({row}, {col}) is outside the {ROWS}x{COLS} grid")
    return row * COLS + col


@dataclass
class TerrainFrame:
    """Local height map in the yaw-aligned ``base_link`` frame.

    Attributes:
        height_m: ``float32[13, 21]``, ``z_base - z_ground`` in metres.
        valid: ``bool[13, 21]``, whether the cell holds a usable measurement.
        stamp_ns: acquisition time of the cloud this frame came from, in
            nanoseconds of a monotonic robot clock (never a wall-clock epoch).
        confidence: ``float32[13, 21]`` in ``[0, 1]``, optional quality signal.
        status: one of the ``STATUS_*`` constants.
        schema_version: layout version; must equal :data:`SCHEMA_VERSION` to be usable.
    """

    height_m: np.ndarray
    valid: np.ndarray
    stamp_ns: int
    confidence: np.ndarray | None = None
    status: str = STATUS_OK
    schema_version: int = SCHEMA_VERSION
    debug: dict[str, Any] = field(default_factory=dict)

    # -- construction ---------------------------------------------------
    def __post_init__(self) -> None:
        self.height_m = np.asarray(self.height_m, dtype=np.float32)
        self.valid = np.asarray(self.valid, dtype=bool)
        if self.confidence is None:
            self.confidence = np.ones((ROWS, COLS), dtype=np.float32)
        else:
            self.confidence = np.asarray(self.confidence, dtype=np.float32)
        self.stamp_ns = int(self.stamp_ns)
        self.validate()

    @classmethod
    def empty(cls, stamp_ns: int = 0, status: str = STATUS_NO_DATA) -> "TerrainFrame":
        """A frame with nothing measured (stale / boot state)."""
        return cls(
            height_m=np.zeros((ROWS, COLS), dtype=np.float32),
            valid=np.zeros((ROWS, COLS), dtype=bool),
            stamp_ns=stamp_ns,
            status=status,
        )

    @classmethod
    def from_height(
        cls,
        height_m: np.ndarray,
        valid: np.ndarray | None = None,
        stamp_ns: int = 0,
        status: str = STATUS_OK,
    ) -> "TerrainFrame":
        """Build a frame from a height map, defaulting ``valid`` to "all finite"."""
        h = np.asarray(height_m, dtype=np.float32)
        if valid is None:
            valid = np.isfinite(h)
        return cls(height_m=h, valid=valid, stamp_ns=stamp_ns, status=status)

    # -- validation -----------------------------------------------------
    def validate(self) -> None:
        if self.height_m.shape != (ROWS, COLS):
            raise TerrainFrameError(
                f"height_m must be {(ROWS, COLS)}, got {self.height_m.shape}"
            )
        if self.valid.shape != (ROWS, COLS):
            raise TerrainFrameError(
                f"valid must be {(ROWS, COLS)}, got {self.valid.shape}"
            )
        if self.confidence.shape != (ROWS, COLS):
            raise TerrainFrameError(
                f"confidence must be {(ROWS, COLS)}, got {self.confidence.shape}"
            )
        if self.status not in _STATUSES:
            raise TerrainFrameError(f"unknown status {self.status!r}")
        if self.schema_version != SCHEMA_VERSION:
            raise TerrainFrameError(
                f"schema_version {self.schema_version} != {SCHEMA_VERSION}"
            )
        finite = np.isfinite(self.height_m)
        if not finite[self.valid].all():
            raise TerrainFrameError("valid cells must have finite heights")

    # -- helpers --------------------------------------------------------
    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK

    def coverage(self) -> float:
        """Fraction of cells that carry a measurement."""
        return float(self.valid.mean())

    def with_status(self, status: str) -> "TerrainFrame":
        clone = self.copy()
        clone.status = status
        clone.validate()
        return clone

    def copy(self) -> "TerrainFrame":
        return TerrainFrame(
            height_m=self.height_m.copy(),
            valid=self.valid.copy(),
            stamp_ns=self.stamp_ns,
            confidence=self.confidence.copy(),
            status=self.status,
            schema_version=self.schema_version,
            debug=dict(self.debug),
        )

    # -- serialisation --------------------------------------------------
    def to_arrays(self) -> dict[str, np.ndarray | int | str]:
        return {
            "schema_version": self.schema_version,
            "rows": ROWS,
            "cols": COLS,
            "resolution_m": RESOLUTION_M,
            "stamp_ns": self.stamp_ns,
            "status": self.status,
            "height_m": self.height_m,
            "valid": self.valid,
            "confidence": self.confidence,
        }

    @classmethod
    def from_arrays(cls, data: dict[str, Any]) -> "TerrainFrame":
        version = int(data.get("schema_version", -1))
        if version != SCHEMA_VERSION:
            raise TerrainFrameError(
                f"replay file schema_version={version}, expected {SCHEMA_VERSION}"
            )
        if int(data["rows"]) != ROWS or int(data["cols"]) != COLS:
            raise TerrainFrameError(
                f"replay grid {data['rows']}x{data['cols']} != {ROWS}x{COLS}"
            )
        return cls(
            height_m=np.asarray(data["height_m"], dtype=np.float32),
            valid=np.asarray(data["valid"], dtype=bool),
            stamp_ns=int(data["stamp_ns"]),
            confidence=np.asarray(data.get("confidence", np.ones((ROWS, COLS)))),
            status=str(data.get("status", STATUS_OK)),
        )
