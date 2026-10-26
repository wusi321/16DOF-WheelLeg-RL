"""Tests for the frozen observation encoding rule."""

import unittest

import numpy as np

from perception.encode import OBS_DIM, decode, encode, encode_many
from perception.frame import CLIP_MAX, CLIP_MIN, COLS, ROWS, TerrainFrame, flat_index


def frame_from(height: np.ndarray, valid: np.ndarray) -> TerrainFrame:
    return TerrainFrame(height_m=height, valid=valid, stamp_ns=0)


class EncodeTest(unittest.TestCase):
    def test_shape(self):
        frame = TerrainFrame.empty()
        self.assertEqual(encode(frame).shape, (OBS_DIM,))
        self.assertEqual(OBS_DIM, ROWS * COLS)

    def test_invalid_becomes_clip_max(self):
        obs = encode(TerrainFrame.empty())
        self.assertTrue(np.all(obs == CLIP_MAX))

    def test_valid_values_are_clipped(self):
        height = np.full((ROWS, COLS), 0.32, dtype=np.float32)
        valid = np.ones((ROWS, COLS), dtype=bool)
        height[0, 0] = -2.0  # ground above the base by 2 m
        height[0, 1] = 3.0  # deep pit
        height[0, 2] = 0.12  # stone top
        obs = encode(frame_from(height, valid))
        self.assertAlmostEqual(float(obs[flat_index(0, 0)]), CLIP_MIN)
        self.assertAlmostEqual(float(obs[flat_index(0, 1)]), CLIP_MAX)
        self.assertAlmostEqual(float(obs[flat_index(0, 2)]), 0.12, places=6)

    def test_layout_is_x_contiguous(self):
        height = np.full((ROWS, COLS), 0.32, dtype=np.float32)
        height[2, 3] = 0.5
        obs = encode(frame_from(height, np.ones((ROWS, COLS), dtype=bool)))
        self.assertAlmostEqual(float(obs[2 * COLS + 3]), 0.5, places=6)
        # neighbour in x is one index further along
        obs[2 * COLS + 3] = 0.32
        self.assertEqual(np.count_nonzero(obs != np.float32(0.32)), 0)

    def test_mix_of_valid_and_invalid(self):
        height = np.full((ROWS, COLS), 0.3, dtype=np.float32)
        valid = np.ones((ROWS, COLS), dtype=bool)
        valid[5, 6] = False
        obs = encode(frame_from(height, valid))
        self.assertAlmostEqual(float(obs[flat_index(5, 6)]), CLIP_MAX)
        self.assertAlmostEqual(float(obs[flat_index(5, 5)]), 0.3, places=6)

    def test_encode_many(self):
        frames = [TerrainFrame.empty(stamp_ns=i) for i in range(3)]
        batch = encode_many(frames)
        self.assertEqual(batch.shape, (3, OBS_DIM))
        self.assertEqual(encode_many([]).shape, (0, OBS_DIM))

    def test_decode_roundtrip(self):
        height = np.full((ROWS, COLS), 0.32, dtype=np.float32)
        height[4, 4] = 0.1
        obs = encode(frame_from(height, np.ones((ROWS, COLS), dtype=bool)))
        grid, valid = decode(obs)
        self.assertEqual(grid.shape, (ROWS, COLS))
        self.assertTrue(valid.all())
        self.assertAlmostEqual(float(grid[4, 4]), 0.1, places=6)


if __name__ == "__main__":
    unittest.main()
