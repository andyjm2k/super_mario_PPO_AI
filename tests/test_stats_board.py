"""Unit tests for the pixel training telemetry board."""

import unittest
import tkinter as tk


class TestStatsBoard(unittest.TestCase):
    """Cover HUD updates without starting a training run."""

    def test_update_stats_and_events(self):
        # Skip when no display is available.
        try:
            # Create a hidden root for widget construction.
            root = tk.Tk()
            # Withdraw so tests stay headless-friendly.
            root.withdraw()
        except tk.TclError:
            # Headless environments cannot build Tk widgets.
            self.skipTest("Tkinter display not available")
        # Import after Tk root exists.
        from ui_stats_panel import StatsBoard

        # Build the telemetry board under the hidden root.
        board = StatsBoard(root)
        # Push a synthetic SB3 dump.
        board.update_stats(
            {
                "rollout/ep_len_mean": 464,
                "rollout/ep_rew_mean": -38.2,
                "time/fps": 810,
                "time/total_timesteps": 65536,
                "train/loss": 9.39,
                "train/approx_kl": 0.011355076,
            }
        )
        # Episode length cell should update.
        self.assertEqual(board.value_vars["rollout/ep_len_mean"].get(), "464")
        # FPS cell should update.
        self.assertEqual(board.value_vars["time/fps"].get(), "810")
        # Loss cell should update to a formatted float.
        self.assertEqual(board.value_vars["train/loss"].get(), "9.39")
        # Event ticker should mention timesteps after the first dump.
        self.assertIn("65536", board.event_var.get())
        # Checkpoint-style events should overwrite the ticker.
        board.set_event("Saved checkpoint: ./train/best_model_4096")
        # Ticker must include the checkpoint path fragment.
        self.assertIn("best_model_4096", board.event_var.get())
        # Destroy the Tk root to free resources.
        root.destroy()


if __name__ == "__main__":
    # Run the stats-board unit tests.
    unittest.main()
