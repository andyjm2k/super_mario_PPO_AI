"""Unit tests for the training dashboard config binding."""

import unittest
import tkinter as tk
from unittest.mock import patch

from training_config import TrainingConfig


class TestTrainingDashboard(unittest.TestCase):
    """Cover dashboard config materialization without starting training."""

    def test_build_config_from_ui_defaults(self):
        # Skip the test when no display is available.
        try:
            # Create a hidden root window for widget variables.
            root = tk.Tk()
            # Withdraw so no window flashes during tests.
            root.withdraw()
        except tk.TclError:
            # Headless CI / servers may not support Tk.
            self.skipTest("Tkinter display not available")
        # Patch train_model import side effects inside the UI module.
        with patch("training_ui.train_model"):
            # Import after patching to keep construction light.
            from training_ui import TrainingDashboard

            # Build the dashboard against the hidden root.
            dashboard = TrainingDashboard(root)
            # Materialize a TrainingConfig from the default UI values.
            config = dashboard._build_config()
            # Defaults should request RecurrentPPO.
            self.assertEqual(config.algorithm, "RecurrentPPO")
            # Defaults should request the CNN+LSTM policy.
            self.assertEqual(config.policy, "CnnLstmPolicy")
            # Defaults should enable live visuals.
            self.assertTrue(config.show_visuals)
            # RecurrentPPO should auto-select a shallow frame stack.
            self.assertEqual(config.frame_stack, 2)
            # Defaults should lean toward larger GPU-friendly batches.
            self.assertGreaterEqual(config.batch_size, 1024)
            # Architecture defaults should expose configurable MLP sizes.
            self.assertEqual(config.pi_arch, "512,512")
            # LSTM hidden size should be present for RecurrentPPO.
            self.assertEqual(config.lstm_hidden_size, 512)
            # Stop button should exist and start disabled.
            self.assertEqual(str(dashboard.stop_btn.cget("state")), "disabled")
            # Config type should be TrainingConfig.
            self.assertIsInstance(config, TrainingConfig)
            # Destroy the Tk root to free resources.
            root.destroy()


if __name__ == "__main__":
    # Run the dashboard unit tests.
    unittest.main()
