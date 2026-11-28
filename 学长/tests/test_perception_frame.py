"""Contract tests for :mod:`perception.frame` (no mjlab, no ROS)."""

import unittest

import numpy as np

from perception.frame import (
    COLS,
    RESOLUTION_M,
    ROWS,
    STATUS_NO_DATA,
    X_MAX,
    X_MIN,
    Y_MAX,
    Y_MIN,
    TerrainFrame,
    TerrainFrameError,
    cell_of,
    flat_index,
    grid_centers,
)


class GridGeometryTest(unittest.TestCase):
    def test_shape(self):
        self.assertEqual((ROWS, COLS), (13, 21))
        self.assertAlmostEqual(RESOLUTION_M, 0.08)

    def test_centers_cover_expected_extent(self):
        xs, ys = grid_centers()
        self.assertEqual(xs.shape, (ROWS, COLS))
        # x is contiguous along the last axis, y along the first.
        self.assertAlmostEqual(float(xs[0, 0]), X_MIN)
        self.assertAlmostEqual(float(xs[0, -1]), X_MAX, places=5)
        self.assertAlmostEqual(float(ys[0, 0]), Y_MIN)
        self.assertAlmostEqual(float(ys[-1, 0]), Y_MAX, places=5)
        # x varies along the row, y varies down the column.
        self.assertGreater(float(xs[0, 1]), float(xs[0, 0]))
        self.assertAlmostEqual(float(xs[0, 1]) - float(xs[0, 0]), RESOLUTION_M, places=5)
        self.assertGreater(float(ys[1, 0]), float(ys[0, 0]))

    def test_cell_of_roundtrip(self):
        xs, ys = grid_centers()
        for row, col in ((0, 0), (5, 7), (12, 20)):
            got = cell_of(float(xs[row, col]), float(ys[row, col]))
            self.assertEqual(got, (row, col))

    def test_cell_of_rejects_outside(self):
        with self.assertRaises(IndexError):
            cell_of(1.0, 0.0)
        with self.assertRaises(IndexError):
            cell_of(0.0, -0.6)

    def test_flat_index_is_x_contiguous(self):
        self.assertEqual(flat_index(0, 0), 0)
        self.assertEqual(flat_index(0, 1), 1)
        self.assertEqual(flat_index(1, 0), COLS)
        with self.assertRaises(IndexError):
            flat_index(ROWS, 0)


class FrameValidationTest(unittest.TestCase):
    def _frame(self, **kwargs):
        height = np.full((ROWS, COLS), 0.32, dtype=np.float32)
        valid = np.ones((ROWS, COLS), dtype=bool)
        params = {"height_m": height, "valid": valid, "stamp_ns": 1}
        params.update(kwargs)
        return TerrainFrame(**params)

    def test_accepts_valid_frame(self):
        frame = self._frame()
        self.assertTrue(frame.ok)
        self.assertEqual(frame.coverage(), 1.0)

    def test_rejects_wrong_shape(self):
        with self.assertRaises(TerrainFrameError):
            self._frame(height_m=np.zeros((12, 21), dtype=np.float32))

    def test_rejects_non_finite_valid_cell(self):
        height = np.full((ROWS, COLS), 0.32, dtype=np.float32)
        height[3, 4] = np.nan
        with self.assertRaises(TerrainFrameError):
            self._frame(height_m=height)

    def test_allows_non_finite_invalid_cell(self):
        height = np.full((ROWS, COLS), 0.32, dtype=np.float32)
        height[3, 4] = np.nan
        valid = np.ones((ROWS, COLS), dtype=bool)
        valid[3, 4] = False
        frame = self._frame(height_m=height, valid=valid)
        self.assertFalse(bool(frame.valid[3, 4]))

    def test_rejects_unknown_status(self):
        with self.assertRaises(TerrainFrameError):
            self._frame(status="bogus")

    def test_rejects_schema_mismatch(self):
        with self.assertRaises(TerrainFrameError):
            self._frame(schema_version=99)

    def test_empty_is_all_invalid(self):
        frame = TerrainFrame.empty(stamp_ns=7)
        self.assertFalse(frame.valid.any())
        self.assertEqual(frame.status, STATUS_NO_DATA)
        self.assertEqual(frame.stamp_ns, 7)

    def test_copy_is_deep_for_arrays(self):
        frame = self._frame()
        clone = frame.copy()
        clone.height_m[0, 0] = 9.0
        self.assertNotEqual(float(frame.height_m[0, 0]), 9.0)

    def test_roundtrip_arrays(self):
        frame = self._frame(stamp_ns=12345)
        restored = TerrainFrame.from_arrays(frame.to_arrays())
        np.testing.assert_array_equal(restored.height_m, frame.height_m)
        np.testing.assert_array_equal(restored.valid, frame.valid)
        self.assertEqual(restored.stamp_ns, 12345)


if __name__ == "__main__":
    unittest.main()
