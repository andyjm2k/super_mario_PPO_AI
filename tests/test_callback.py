"""Unit tests for the training callback stop + checkpoint helpers."""

import os
import tempfile
import threading
import unittest
from unittest.mock import MagicMock

from callback import TrainAndLoggingCallback


class TestTrainAndLoggingCallback(unittest.TestCase):
    """Cover checkpoint path creation, best-progress, and cooperative stop."""

    def test_init_callback_creates_save_path(self):
        # Use a temporary directory for checkpoint output.
        with tempfile.TemporaryDirectory() as tmp:
            # Nested path that does not exist yet.
            save_path = os.path.join(tmp, "checkpoints")
            # Build the callback with a short check frequency.
            callback = TrainAndLoggingCallback(check_freq=10, save_path=save_path)
            # Initialize directories via the SB3 hook.
            callback._init_callback()
            # Save path must exist afterward.
            self.assertTrue(os.path.isdir(save_path))

    def test_maybe_save_best_progress(self):
        # Use a temporary directory for the best-progress model.
        with tempfile.TemporaryDirectory() as tmp:
            # Build the callback pointed at the temp directory.
            callback = TrainAndLoggingCallback(check_freq=10, save_path=tmp, verbose=0)
            # Fake vectorized env exposing last_info on workers.
            fake_env = MagicMock()
            # Two workers reporting progress percentages.
            fake_env.get_attr.return_value = [
                {"progress_pct": 10.0},
                {"progress_pct": 30.0},
            ]
            # Fake model with save + get_env for the training_env property.
            callback.model = MagicMock()
            # SB3 resolves training_env through model.get_env().
            callback.model.get_env.return_value = fake_env
            # Trigger the best-progress saver.
            callback._maybe_save_best_progress()
            # Mean progress should become the new best (20).
            self.assertEqual(callback.best_mean_progress, 20.0)
            # Model save should have been called once.
            callback.model.save.assert_called_once()

    def test_stop_event_halts_training(self):
        # Use a temporary directory for the stop snapshot.
        with tempfile.TemporaryDirectory() as tmp:
            # Create a stop event already set (as if the UI pressed Stop).
            stop_event = threading.Event()
            # Arm the stop flag before the callback step.
            stop_event.set()
            # Build the callback with the armed stop event.
            callback = TrainAndLoggingCallback(
                check_freq=10, save_path=tmp, verbose=0, stop_event=stop_event
            )
            # Fake model used when saving stopped_model.
            callback.model = MagicMock()
            # _on_step must return False to end model.learn().
            self.assertFalse(callback._on_step())
            # The callback should record that the user requested the stop.
            self.assertTrue(callback.stopped_by_user)
            # A stop snapshot should have been requested.
            callback.model.save.assert_called()

    def test_on_training_start_attaches_stats_writer(self):
        # Fake queue that will receive logger dumps.
        fake_queue = MagicMock()
        # Build a callback that should install QueueStatsWriter.
        callback = TrainAndLoggingCallback(
            check_freq=10, save_path=".", verbose=0, frame_queue=fake_queue
        )
        # Fake model logger with an empty output_formats list.
        fake_logger = MagicMock()
        # Start with no writers installed.
        fake_logger.output_formats = []
        # Fake model exposing logger for BaseCallback.logger.
        callback.model = MagicMock()
        # Point model.logger at our controllable fake logger.
        callback.model.logger = fake_logger
        # Run the training-start hook.
        callback._on_training_start()
        # Exactly one QueueStatsWriter should have been appended.
        self.assertEqual(len(fake_logger.output_formats), 1)
        # Import the writer class for the isinstance check.
        from stats_writer import QueueStatsWriter

        # Confirm the installed writer type.
        self.assertIsInstance(fake_logger.output_formats[0], QueueStatsWriter)
        # Calling again must not duplicate the writer.
        callback._on_training_start()
        # Still exactly one writer.
        self.assertEqual(len(fake_logger.output_formats), 1)


if __name__ == "__main__":
    # Run the callback unit tests.
    unittest.main()
