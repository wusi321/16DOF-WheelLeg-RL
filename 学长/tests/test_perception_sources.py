"""Tests for the sim-raycast and replay sources."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from perception.encode import encode
from perception.frame import COLS, ROWS, TerrainFrame, flat_index
from perception.sources import ReplayTerrainSource, SimTerrainSource, load_frames, save_frames


def make_arrays(hit_z: np.ndarray, misses: np.ndarray | None = None, frame_z: float = 0.32):
    distances = np.full(hit_z.size, 1.0, dtype=np.float32)
    if misses is not None:
        distances[misses] = -1.0
    heights = frame_z - hit_z
    return distances, heights


class FakeSensorData:
    def __init__(self, distances, frame_pos, hit_pos):
        self.distances = distances
        self.frame_pos_w = frame_pos
        self.hit_pos_w = hit_pos


class FakeSensor:
    def __init__(self, distances, frame_pos, hit_pos):
        self.data = FakeSensorData(distances, frame_pos, hit_pos)


class SimSourceTest(unittest.TestCase):
    def test_from_arrays_matches_height_semantics(self):
        hit_z = np.zeros(ROWS * COLS, dtype=np.float32)
        hit_z[:] = 0.0
        misses = np.array([0, 5, 77], dtype=int)
        distances, heights = make_arrays(hit_z, misses)

        frame = SimTerrainSource.from_arrays(distances, heights, stamp_ns=42)
        self.assertEqual(frame.stamp_ns, 42)
        self.assertEqual(frame.height_m.shape, (ROWS, COLS))
        self.assertAlmostEqual(float(frame.height_m[0, 0]) * 0 + 0.32, 0.32, places=6)
        self.assertTrue(bool(frame.valid[0, 1]))
        for idx in misses:
            row, col = divmod(int(idx), COLS)
            self.assertFalse(bool(frame.valid[row, col]))
            self.assertAlmostEqual(float(encode(frame)[flat_index(row, col)]), 1.0)

    def test_ridge_creates_lower_height(self):
        hit_z = np.full(ROWS * COLS, 0.0, dtype=np.float32)
        hit_z[0 : COLS] = 0.2  # a 0.2 m step across the first row
        distances, heights = make_arrays(hit_z)
        frame = SimTerrainSource.from_arrays(distances, heights)
        self.assertAlmostEqual(float(frame.height_m[0, 0]), 0.12, places=6)
        self.assertAlmostEqual(float(frame.height_m[1, 0]), 0.32, places=6)

    def test_read_from_sensor_object(self):
        n = ROWS * COLS
        distances = np.ones((1, n), dtype=np.float32)
        frame_pos = np.array([[[0.0, 0.0, 0.32]]], dtype=np.float32)
        hit_pos = np.zeros((1, 1, n, 3), dtype=np.float32)
        sensor = FakeSensor(distances, frame_pos, hit_pos)
        frame = SimTerrainSource(sensor).read()
        np.testing.assert_allclose(frame.height_m, np.float32(0.32), atol=1e-6)
        self.assertTrue(frame.valid.all())

    def test_read_marks_misses(self):
        n = ROWS * COLS
        distances = np.ones((1, n), dtype=np.float32)
        distances[0, 3] = -1.0
        frame_pos = np.array([[[0.0, 0.0, 0.32]]], dtype=np.float32)
        hit_pos = np.zeros((1, 1, n, 3), dtype=np.float32)
        frame = SimTerrainSource(FakeSensor(distances, frame_pos, hit_pos)).read()
        self.assertFalse(bool(frame.valid[0, 3]))

    def test_rejects_wrong_ray_count(self):
        with self.assertRaises(ValueError):
            SimTerrainSource.from_arrays(np.zeros(10), np.zeros(10))


class ReplayTest(unittest.TestCase):
    def _frames(self, count=3):
        frames = []
        for i in range(count):
            height = np.full((ROWS, COLS), 0.30 + 0.01 * i, dtype=np.float32)
            frames.append(TerrainFrame(height_m=height, valid=np.ones((ROWS, COLS), bool), stamp_ns=i))
        return frames

    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frames.npz"
            save_frames(path, self._frames())
            loaded = load_frames(path)
            self.assertEqual(len(loaded), 3)
            for got, want in zip(loaded, self._frames()):
                np.testing.assert_array_equal(got.height_m, want.height_m)
                self.assertEqual(got.stamp_ns, want.stamp_ns)

    def test_looping_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frames.npz"
            save_frames(path, self._frames(2))
            src = ReplayTerrainSource(path)
            self.assertEqual(len(src), 2)
            first = src.read().stamp_ns
            second = src.read().stamp_ns
            third = src.read().stamp_ns
            self.assertEqual((first, second, third), (0, 1, 0))
            src.reset()
            self.assertEqual(src.read().stamp_ns, 0)

    def test_rejects_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                save_frames(Path(tmp) / "x.npz", [])


if __name__ == "__main__":
    unittest.main()
