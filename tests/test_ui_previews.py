"""Unit tests for live agent preview helpers."""

import unittest

from ui_previews import preview_size


class TestPreviewSize(unittest.TestCase):
    """Cover preview_size aspect fitting for GB and GBA frames."""

    def test_game_boy_native(self):
        # GBC 160x144 should remain 160x144 inside the booth box.
        width, height = preview_size(144, 160)
        # Width matches native.
        self.assertEqual(width, 160)
        # Height matches native.
        self.assertEqual(height, 144)

    def test_gba_native(self):
        # GBA 240x160 should remain 240x160.
        width, height = preview_size(160, 240)
        # Width matches native.
        self.assertEqual(width, 240)
        # Height matches native.
        self.assertEqual(height, 160)

    def test_rejects_zero_dims(self):
        # Zero inputs still produce at least 1x1.
        width, height = preview_size(0, 0)
        # Minimum width.
        self.assertGreaterEqual(width, 1)
        # Minimum height.
        self.assertGreaterEqual(height, 1)


if __name__ == "__main__":
    # Run preview helper unit tests.
    unittest.main()
