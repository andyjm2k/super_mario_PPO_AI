"""Unit tests for the model runner helpers and dashboard wiring."""

import os
import tempfile
import unittest
from unittest.mock import patch

import tkinter as tk


class TestListCheckpoints(unittest.TestCase):
    """Cover checkpoint discovery used by the runner UI."""

    def test_list_checkpoints_finds_zip(self):
        # Import the discovery helper.
        from run_model import list_checkpoints

        # Use a temporary train directory.
        with tempfile.TemporaryDirectory() as tmp:
            # Create a fake checkpoint zip file.
            zip_path = os.path.join(tmp, "best_progress_model.zip")
            # Write a tiny placeholder file.
            with open(zip_path, "wb") as handle:
                # Minimal content so the file exists.
                handle.write(b"zip")
            # Discover checkpoints in the temp folder.
            found = list_checkpoints(tmp)
            # The fake zip must be listed.
            self.assertIn(zip_path, found)


class TestRunnerDashboard(unittest.TestCase):
    """Cover runner dashboard construction without starting evaluation."""

    def test_runner_builds_and_lists_defaults(self):
        # Skip when no display is available.
        try:
            # Create a hidden root window.
            root = tk.Tk()
            # Withdraw so tests stay headless-friendly.
            root.withdraw()
        except tk.TclError:
            # Headless environments cannot build Tk widgets.
            self.skipTest("Tkinter display not available")
        # Patch episode running so construction stays light.
        with patch("runner_ui.run_episodes"):
            # Import after patching.
            from runner_ui import ModelRunnerDashboard

            # Build the runner against the hidden root.
            dashboard = ModelRunnerDashboard(root)
            # Defaults should request RecurrentPPO.
            self.assertEqual(dashboard.vars["algorithm"].get(), "RecurrentPPO")
            # Stop should start disabled.
            self.assertEqual(str(dashboard.stop_btn.cget("state")), "disabled")
            # Destroy the Tk root to free resources.
            root.destroy()


if __name__ == "__main__":
    # Run the runner unit tests.
    unittest.main()
