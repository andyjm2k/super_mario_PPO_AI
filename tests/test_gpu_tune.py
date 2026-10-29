"""Unit tests for GPU tuning helpers and presets."""

import unittest

from gpu_tune import suggest_gpu_hyperparams, validate_batch_divides_buffer


class TestGpuTune(unittest.TestCase):
    """Cover preset geometry and batch/buffer validation."""

    def test_recurrent_preset_divides_cleanly(self):
        # Load the RecurrentPPO GPU preset.
        preset = suggest_gpu_hyperparams("RecurrentPPO")
        # Buffer size must be divisible by batch_size.
        buffer_size = preset["n_envs"] * preset["n_steps"]
        # Assert exact divisibility.
        self.assertEqual(buffer_size % preset["batch_size"], 0)

    def test_ppo_preset_divides_cleanly(self):
        # Load the feed-forward PPO GPU preset.
        preset = suggest_gpu_hyperparams("PPO")
        # Buffer size must be divisible by batch_size.
        buffer_size = preset["n_envs"] * preset["n_steps"]
        # Assert exact divisibility.
        self.assertEqual(buffer_size % preset["batch_size"], 0)

    def test_validate_batch_divides_buffer_rejects_bad_geometry(self):
        # 8 * 512 = 4096 is not divisible by 1000.
        error = validate_batch_divides_buffer(8, 512, 1000, "RecurrentPPO")
        # An error string must be returned.
        self.assertIsNotNone(error)

    def test_validate_batch_divides_buffer_accepts_good_geometry(self):
        # 16 * 1024 = 16384 is divisible by 2048.
        error = validate_batch_divides_buffer(16, 1024, 2048, "RecurrentPPO")
        # Compatible configs return None.
        self.assertIsNone(error)

    def test_qrdqn_skips_buffer_validation(self):
        # QR-DQN does not use the on-policy buffer constraint.
        error = validate_batch_divides_buffer(3, 7, 11, "QRDQN")
        # Validation should be skipped.
        self.assertIsNone(error)


if __name__ == "__main__":
    # Run the GPU-tune unit tests.
    unittest.main()
