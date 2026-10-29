"""Unit tests for hover tooltip helpers."""

import unittest
import tkinter as tk

from ui_tooltips import MODEL_TOOLTIPS, TRAIN_TOOLTIPS, HoverTooltip, attach_tooltip


class TestTooltips(unittest.TestCase):
    """Cover tooltip dictionaries and attachment wiring."""

    def test_train_tooltips_cover_core_keys(self):
        # Core TRAIN controls must have explanations.
        for key in ("ALGORITHM", "N_STEPS", "BATCH", "PARALLEL ENVS"):
            # Each key should map to a non-empty tip string.
            self.assertTrue(TRAIN_TOOLTIPS[key].strip())

    def test_model_tooltips_cover_core_keys(self):
        # Core MODEL controls must have explanations.
        for key in ("PI LAYERS", "VF LAYERS", "LSTM HIDDEN", "FEATURES DIM"):
            # Each key should map to a non-empty tip string.
            self.assertTrue(MODEL_TOOLTIPS[key].strip())

    def test_attach_tooltip_binds_widget(self):
        # Skip when no display is available.
        try:
            # Create a hidden root for widget construction.
            root = tk.Tk()
            # Withdraw so tests stay headless-friendly.
            root.withdraw()
        except tk.TclError:
            # Headless environments cannot build Tk widgets.
            self.skipTest("Tkinter display not available")
        # Build a simple label to attach a tip to.
        label = tk.Label(root, text="TEST")
        # Attach a tooltip instance.
        tip = attach_tooltip(label, "Hello tip")
        # The helper should return a HoverTooltip.
        self.assertIsInstance(tip, HoverTooltip)
        # Text payload should be stored on the tip.
        self.assertEqual(tip.text, "Hello tip")
        # Destroy the root to free resources.
        root.destroy()


if __name__ == "__main__":
    # Run the tooltip unit tests.
    unittest.main()
