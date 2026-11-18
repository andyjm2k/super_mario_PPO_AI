"""Aspect checks for the policy frame and the booth picture."""

import unittest

import numpy as np

from pdt.emulator.mame_frame import bgra_to_rgb, display_bgra, letterbox


def _bgra(width, height, pixel):
    """Repeat one BGRA pixel into a packed screen dump."""
    # One pixel, four bytes.
    return bytes(pixel) * (int(width) * int(height))


class TestMameFrameAspect(unittest.TestCase):
    """The training tensor stays full-bleed. The booth picture does not stretch."""

    def test_policy_frame_fills_128x96(self):
        # A wide blue screen.
        raw = _bgra(8, 2, (255, 0, 0, 255))
        # Policy path.
        frame = bgra_to_rgb(raw, 8, 2)
        # Fixed CNN size.
        self.assertEqual(frame.shape, (96, 128, 3))
        # The corner is picture, not a black bar.
        self.assertGreater(int(frame[0, 0, 2]), 200)

    def test_booth_keeps_wide_aspect(self):
        # Same wide blue screen.
        raw = _bgra(8, 2, (255, 0, 0, 255))
        # Booth path.
        glass = display_bgra(raw, 8, 2)
        # Booth box.
        self.assertEqual(glass.shape, (160, 240, 3))
        # Top padding stays black.
        self.assertEqual(int(glass[0, 0, 0]), 0)
        # The picture sits in a horizontal band.
        self.assertGreater(int(glass[80, 120, 2]), 200)

    def test_arcade_ratio_inside_the_glass(self):
        # CPS3 is 384x224, about 12:7, not 4:3.
        rgb = np.full((224, 384, 3), 255, dtype=np.uint8)
        # Fit into the booth.
        glass = letterbox(rgb, 384, 224, 240, 160)
        # Find the white rows.
        white = np.where(glass[:, 0, 0] > 200)[0]
        # There is a picture.
        self.assertGreater(len(white), 0)
        # Height of that picture.
        picture_h = int(white[-1] - white[0] + 1)
        # Width is the full glass because 12:7 is wider than 240:160.
        picture_w = 240
        # Restored ratio stays near 384/224.
        self.assertAlmostEqual(picture_w / picture_h, 384 / 224, places=1)


if __name__ == "__main__":
    # Run the frame aspect tests.
    unittest.main()
