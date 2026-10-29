"""Unit tests for training configuration helpers."""

import os
import tempfile
import unittest

from training_config import TrainingConfig


class TestTrainingConfig(unittest.TestCase):
    """Verify directory helpers and start-state discovery."""

    def test_ensure_dirs_creates_folders(self):
        # Use a temporary directory so tests do not touch real train/logs.
        with tempfile.TemporaryDirectory() as tmp:
            # Point checkpoints and logs into the temp folder.
            config = TrainingConfig(
                checkpoint_dir=os.path.join(tmp, "train"),
                log_dir=os.path.join(tmp, "logs"),
            )
            # Create the directories.
            config.ensure_dirs()
            # Checkpoint dir must exist afterward.
            self.assertTrue(os.path.isdir(config.checkpoint_dir))
            # Log dir must exist afterward.
            self.assertTrue(os.path.isdir(config.log_dir))

    def test_resolve_state_paths_explicit(self):
        # Create temporary fake state files.
        with tempfile.TemporaryDirectory() as tmp:
            # First fake state path.
            state_a = os.path.join(tmp, "a.state")
            # Second fake state path.
            state_b = os.path.join(tmp, "b.state")
            # Create empty state files on disk.
            open(state_a, "wb").close()
            # Create the second empty state file.
            open(state_b, "wb").close()
            # Configure explicit states including one missing path.
            config = TrainingConfig(state_paths=[state_a, state_b, os.path.join(tmp, "missing.state")])
            # Resolve should drop the missing path.
            resolved = config.resolve_state_paths()
            # Both real files should remain.
            self.assertEqual(set(resolved), {state_a, state_b})

    def test_default_algorithm_is_recurrent_ppo(self):
        # Defaults should prefer RecurrentPPO for multi-level generalization.
        config = TrainingConfig()
        # Algorithm default must be RecurrentPPO.
        self.assertEqual(config.algorithm, "RecurrentPPO")
        # Policy default must be the CNN+LSTM policy.
        self.assertEqual(config.policy, "CnnLstmPolicy")


if __name__ == "__main__":
    # Run the training-config unit tests.
    unittest.main()
