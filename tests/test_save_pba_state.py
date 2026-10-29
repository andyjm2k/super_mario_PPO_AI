"""Unit tests for the GBA snapshot helper key maps."""

import unittest

import sdl2

from tools.save_pba_state import build_key_maps
from pyboy_advance.constants import Key


class TestBuildKeyMaps(unittest.TestCase):
    """Cover build_key_maps button bindings used by the snapshot tool."""

    def test_maps_core_buttons(self):
        # Build both key maps.
        key_down, key_up = build_key_maps()
        # A / B face buttons.
        self.assertEqual(key_down[sdl2.SDLK_a], Key.BUTTON_A)
        # B is on S.
        self.assertEqual(key_down[sdl2.SDLK_s], Key.BUTTON_B)
        # Start / select.
        self.assertEqual(key_down[sdl2.SDLK_RETURN], Key.BUTTON_START)
        # Select on backspace.
        self.assertEqual(key_down[sdl2.SDLK_BACKSPACE], Key.BUTTON_SELECT)

    def test_maps_dpad_and_shoulders(self):
        # Build both key maps.
        key_down, key_up = build_key_maps()
        # D-pad directions.
        self.assertEqual(key_down[sdl2.SDLK_UP], Key.DPAD_UP)
        # Left.
        self.assertEqual(key_down[sdl2.SDLK_LEFT], Key.DPAD_LEFT)
        # Shoulders.
        self.assertEqual(key_down[sdl2.SDLK_q], Key.SHOULDER_LEFT)
        # Right shoulder.
        self.assertEqual(key_down[sdl2.SDLK_w], Key.SHOULDER_RIGHT)

    def test_keyup_matches_keydown(self):
        # Build both key maps.
        key_down, key_up = build_key_maps()
        # Release map should mirror press map.
        self.assertEqual(key_down, key_up)


if __name__ == "__main__":
    # Run snapshot helper unit tests.
    unittest.main()
