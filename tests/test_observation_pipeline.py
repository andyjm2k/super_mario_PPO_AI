"""Unit tests for observation preprocessing helpers."""

import unittest

import numpy as np

from observation_pipeline import preprocess_frame, resolve_frame_stack


class TestObservationPipeline(unittest.TestCase):
    """Cover HUD crop, resize shape, and stack-depth defaults."""

    def test_preprocess_frame_shape(self):
        # Build a fake native GBC RGB frame (144x160x3).
        rgb = np.zeros((144, 160, 3), dtype=np.uint8)
        # Paint the HUD rows bright so a bad crop would keep them.
        rgb[:16, :, :] = 255
        # Paint the playfield a mid gray for visibility.
        rgb[16:, :, :] = 128
        # Run the standard preprocess pipeline.
        obs = preprocess_frame(rgb, crop_top=16, obs_width=80, obs_height=64)
        # Output must be a single-channel CNN frame.
        self.assertEqual(obs.shape, (64, 80, 1))
        # Dtype must stay uint8 for SB3 image policies.
        self.assertEqual(obs.dtype, np.uint8)
        # After cropping the white HUD, the frame should not be all-white.
        self.assertLess(float(obs.max()), 255.0)

    def test_preprocess_rejects_bad_shape(self):
        # A grayscale plane is not a valid RGB input.
        bad = np.zeros((144, 160), dtype=np.uint8)
        # Expect a clear ValueError.
        with self.assertRaises(ValueError):
            # Call preprocess with an invalid array.
            preprocess_frame(bad)

    def test_resolve_frame_stack_defaults(self):
        # RecurrentPPO should use a shallow stack.
        self.assertEqual(resolve_frame_stack("RecurrentPPO"), 2)
        # PPO should use a classic 4-frame stack.
        self.assertEqual(resolve_frame_stack("PPO"), 4)
        # QRDQN should also use a 4-frame stack.
        self.assertEqual(resolve_frame_stack("QRDQN"), 4)

    def test_resolve_frame_stack_override(self):
        # Explicit overrides must win over algorithm defaults.
        self.assertEqual(resolve_frame_stack("RecurrentPPO", 3), 3)


if __name__ == "__main__":
    # Run the observation pipeline unit tests.
    unittest.main()
