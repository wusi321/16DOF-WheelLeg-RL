"""Tests for the pure-numpy stepping-stone touch geometry."""

import unittest

import numpy as np

from perception.stones import (
    TOUCH_XY_TOL,
    TOUCH_Z_TOL,
    new_touches,
    pad_stone_table,
    touched_stones,
)


class TouchGeometryTest(unittest.TestCase):
    def setUp(self):
        # One 0.2 x 0.2 stone with its top at 0.10 m.
        self.stone_x = np.array([0.0])
        self.stone_y = np.array([0.0])
        self.half_x = np.array([0.10])
        self.half_y = np.array([0.10])
        self.top = np.array([0.10])

    def _touch(self, wheel_xy, wheel_bottom):
        return touched_stones(
            self.stone_x,
            self.stone_y,
            self.half_x,
            self.half_y,
            self.top,
            np.asarray(wheel_xy, dtype=float).reshape(-1, 2),
            np.asarray(wheel_bottom, dtype=float).reshape(-1),
        )

    def test_wheel_on_stone(self):
        self.assertTrue(self._touch([[0.0, 0.0]], [0.10])[0])

    def test_wheel_near_edge_counts(self):
        # Inside the footprint plus the tolerance.
        self.assertTrue(self._touch([[0.12, 0.0]], [0.10])[0])

    def test_wheel_outside_is_ignored(self):
        self.assertFalse(self._touch([[0.20, 0.0]], [0.10])[0])

    def test_wheel_too_high_is_not_contact(self):
        self.assertFalse(self._touch([[0.0, 0.0]], [0.10 + TOUCH_Z_TOL + 0.01])[0])

    def test_wheel_too_low_is_not_contact(self):
        self.assertFalse(self._touch([[0.0, 0.0]], [0.10 - TOUCH_Z_TOL - 0.01])[0])

    def test_any_wheel_counts(self):
        mask = self._touch([[0.5, 0.5], [0.0, 0.0]], [0.30, 0.10])
        self.assertTrue(mask[0])

    def test_multiple_stones(self):
        mask = touched_stones(
            np.array([0.0, 0.6]),
            np.array([0.0, 0.0]),
            np.array([0.10, 0.10]),
            np.array([0.10, 0.10]),
            np.array([0.10, 0.20]),
            np.array([[0.0, 0.0]]),
            np.array([0.10]),
        )
        self.assertTrue(mask[0])
        self.assertFalse(mask[1])

    def test_tolerance_constant_is_small(self):
        # A 3 cm margin must not bridge the 40 cm gap between two lanes.
        self.assertLess(TOUCH_XY_TOL, 0.20)


class BookkeepingTest(unittest.TestCase):
    def test_new_touches(self):
        visited = np.array([True, False, False])
        touched = np.array([True, True, False])
        np.testing.assert_array_equal(new_touches(visited, touched), [False, True, False])

    def test_padding_rows_never_touched(self):
        table, valid = pad_stone_table([(0.0, 0.0, 0.1, 0.1, 0.1)], max_stones=3)
        self.assertEqual(table.shape, (3, 5))
        self.assertEqual(valid.tolist(), [True, False, False])
        mask = touched_stones(
            table[:, 0], table[:, 1], table[:, 2], table[:, 3], table[:, 4],
            np.array([[0.0, 0.0]]), np.array([0.1]),
        )
        self.assertEqual(mask.tolist(), [True, False, False])


if __name__ == "__main__":
    unittest.main()
