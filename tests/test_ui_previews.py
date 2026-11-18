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

    def test_wide_art_is_not_stretched(self):
        # A 1000x400 painting fitted into the old 1040x168 banner slot.
        from gb_theme import fit_size

        # The old code forced 1040x168 and squashed this picture.
        width, height = fit_size(1000, 400, 1040, 168)
        # Height is the limiting side.
        self.assertEqual(height, 168)
        # Width follows the source ratio instead of filling 1040.
        self.assertEqual(width, 420)
        # A square sticker fitted into a tall strip stays square.
        sticker_w, sticker_h = fit_size(500, 500, 120, 168)
        # Both sides match.
        self.assertEqual((sticker_w, sticker_h), (120, 120))

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
