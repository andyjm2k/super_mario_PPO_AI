"""Unit tests for the link-window button hold latch."""

import unittest

from tools.link_setup import InputLatch


class TestInputLatch(unittest.TestCase):
    """A short tap must stay down long enough for character select."""

    def test_tap_stays_down_after_release(self):
        # Three frames is enough to see the hold without the full game.
        latch = InputLatch(hold_frames=3)
        # Player 1 taps right.
        latch.note_down(0, "right")
        # The finger is already up.
        latch.note_up(0, "right")
        # The core should still see the button.
        self.assertIn("right", latch.held(0))
        # One frame passes.
        latch.age()
        # Still held.
        self.assertIn("right", latch.held(0))
        # Two more frames and the tap expires.
        latch.age()
        latch.age()
        # Gone.
        self.assertNotIn("right", latch.held(0))

    def test_held_key_survives_the_timer(self):
        # A key that stays down must not release when the tap timer ends.
        latch = InputLatch(hold_frames=1)
        # Held, not a tap.
        latch.note_down(1, "a")
        # Timer expires.
        latch.age()
        # The physical key is still down.
        self.assertIn("a", latch.held(1))
        # Player 1 is unaffected.
        self.assertNotIn("a", latch.held(0))

    def test_players_do_not_share_buttons(self):
        # Arrow keys must not leak onto the other GBA.
        latch = InputLatch(hold_frames=4)
        # Left side only.
        latch.note_down(0, "left")
        # Right side stays idle.
        self.assertEqual(latch.held(1), set())
        # Left side has the button.
        self.assertEqual(latch.held(0), {"left"})


if __name__ == "__main__":
    # Run the latch tests.
    unittest.main()
