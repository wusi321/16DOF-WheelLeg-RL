"""Tests for the sim2real randomisation helpers."""

import unittest

import numpy as np

from perception.augment import (
    DelayedSource,
    add_range_noise,
    apply_static_mask,
    make_rect_mask,
    random_dropout,
    smooth,
)
from perception.frame import COLS, ROWS, X_MIN, TerrainFrame


def frame(height=0.32, valid=True):
    return TerrainFrame(
        height_m=np.full((ROWS, COLS), height, dtype=np.float32),
        valid=np.full((ROWS, COLS), valid, dtype=bool),
        stamp_ns=0,
    )


class DummySource:
    def __init__(self, stamps):
        self.stamps = list(stamps)
        self.i = 0

    def read(self):
        stamp = self.stamps[self.i % len(self.stamps)]
        self.i += 1
        return TerrainFrame(
            height_m=np.full((ROWS, COLS), 0.32, dtype=np.float32),
            valid=np.ones((ROWS, COLS), dtype=bool),
            stamp_ns=stamp,
        )


class NoiseTest(unittest.TestCase):
    def test_reproducible_with_same_seed(self):
        a = add_range_noise(frame(), np.random.default_rng(0))
        b = add_range_noise(frame(), np.random.default_rng(0))
        np.testing.assert_array_equal(a.height_m, b.height_m)

    def test_does_not_touch_invalid_cells(self):
        f = frame()
        f.valid[:] = False
        f.height_m[:] = 1.0
        out = add_range_noise(f, np.random.default_rng(0))
        np.testing.assert_array_equal(out.height_m, np.float32(1.0))

    def test_sigma_grows_with_distance(self):
        near = frame(height=0.2)
        far = frame(height=5.0)
        rng = np.random.default_rng(1)
        n = add_range_noise(near, rng)
        f = add_range_noise(far, rng)
        self.assertLess(float(np.std(n.height_m - near.height_m)),
                        float(np.std(f.height_m - far.height_m)))


class DropoutTest(unittest.TestCase):
    def test_band_probabilities(self):
        close = random_dropout(frame(height=0.5), np.random.default_rng(0),
                               bands=((1.0, 0.0), (np.inf, 1.0)))
        self.assertTrue(close.valid.all())
        far = random_dropout(frame(height=2.0), np.random.default_rng(0),
                             bands=((1.0, 0.0), (np.inf, 1.0)))
        self.assertFalse(far.valid.any())
        self.assertEqual(far.status, "all_invalid")

    def test_never_revalidates_dropped_cells(self):
        f = TerrainFrame(
            height_m=np.full((ROWS, COLS), 0.3, dtype=np.float32),
            valid=np.zeros((ROWS, COLS), dtype=bool),
            stamp_ns=0,
        )
        out = random_dropout(f, np.random.default_rng(0))
        self.assertFalse(out.valid.any())


class MaskTest(unittest.TestCase):
    def test_rect_mask_blank(self):
        mask = make_rect_mask((-0.2, 0.2), (-0.1, 0.1))
        self.assertEqual(mask.shape, (ROWS, COLS))
        self.assertTrue(mask[:, 10].any())
        self.assertFalse(mask[:, 0].any())

    def test_apply_mask(self):
        mask = make_rect_mask((X_MIN, X_MIN + 0.08), (-1.0, 1.0))
        out = apply_static_mask(frame(), mask)
        self.assertTrue(out.valid[:, 0].all())
        self.assertFalse(out.valid[:, 5].any())

    def test_mask_shape_checked(self):
        with self.assertRaises(ValueError):
            apply_static_mask(frame(), np.zeros((3, 3), dtype=bool))


class SmoothTest(unittest.TestCase):
    def test_removes_single_spike(self):
        f = frame()
        f.height_m[5, 5] = 2.0
        out = smooth(f)
        self.assertLess(float(out.height_m[5, 5]), 1.0)

    def test_rejects_even_kernel(self):
        with self.assertRaises(ValueError):
            smooth(frame(), kernel=4)


class DelayedSourceTest(unittest.TestCase):
    def test_first_read_is_stale(self):
        src = DelayedSource(DummySource([0, 1, 2]), delay_steps=1)
        first = src.read()
        self.assertEqual(first.status, "stale")
        second = src.read()
        self.assertEqual(second.status, "ok")
        self.assertEqual(second.stamp_ns, 0)

    def test_zero_delay_passes_through(self):
        src = DelayedSource(DummySource([7]), delay_steps=0)
        self.assertEqual(src.read().stamp_ns, 7)

    def test_negative_delay_rejected(self):
        with self.assertRaises(ValueError):
            DelayedSource(DummySource([0]), delay_steps=-1)


if __name__ == "__main__":
    unittest.main()
