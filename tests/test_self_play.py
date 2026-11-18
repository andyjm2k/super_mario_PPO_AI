"""Unit tests for GBA link self-play helpers."""

import unittest

from pdt.emulator.gba_link import _CableSync, link_symbols_available
from pdt.game_pack import load_game_pack
from pdt.self_play_env import SelfPlayEnv


class TestCableSync(unittest.TestCase):
    """Cover the in-process stand-in for mGBA's lockstep wait/signal."""

    def test_signal_then_wait(self):
        # wait() only parks the master. signal() is what lets it run again.
        sync = _CableSync(2)
        # mGBA aborts if wait() returns false, and it must return immediately.
        self.assertTrue(sync.wait(0b10))
        # The master is parked until the slave bit is cleared.
        self.assertLess(sync.awake[0], 1)
        # Slave reaches the transfer.
        woke = sync.signal(0b10)
        # The parked master was released.
        self.assertTrue(woke)
        # The scheduler may step the master again.
        self.assertGreaterEqual(sync.awake[0], 1)

    def test_add_cycles_wakes_a_parked_slave(self):
        # A slave that spends cycles it does not have must park, not block the thread.
        sync = _CableSync(2)
        # Spending from an empty pool goes negative and parks player 2.
        left = sync.use_cycles(1, 5)
        # The C driver uses a negative return as an early-exit.
        self.assertLess(left, 0)
        # Player 2 must not be stepped until the master posts more.
        self.assertLess(sync.awake[1], 1)
        # Master posts a fresh slice. That is the wake.
        sync.add_cycles(0, 100)
        # The pool is the grant minus what was already spent.
        self.assertEqual(sync.posted[1], 95)
        # Player 2 can run again.
        self.assertGreaterEqual(sync.awake[1], 1)

    def test_add_and_use_cycles(self):
        # Two-player cable.
        sync = _CableSync(2)
        # Master posts 100 cycles, which credits the slave.
        sync.add_cycles(0, 100)
        # Slave still has the full credit.
        self.assertEqual(sync.unused_cycles(1), 100)
        # Spending 40 leaves 60.
        self.assertEqual(sync.use_cycles(1, 40), 60)


class TestSelfPlayGuard(unittest.TestCase):
    """Cover pack checks that happen before any emulator boots."""

    def test_street_fighter_opts_into_link(self):
        # The GBA fighting pack declares link versus.
        pack = load_game_pack("street_fighter3")
        # UI self-play is allowed for this pack.
        self.assertTrue(pack.link_play)

    def test_mario_is_not_a_link_game(self):
        # Single-player packs must fail before a cable is created.
        with self.assertRaises(ValueError):
            # Constructor checks link_play first.
            SelfPlayEnv(game_id="mario_deluxe")

    def test_link_pair_names_round_trip(self):
        # A saved pair should be discovered and hidden from solo starts.
        import os
        import tempfile
        from pdt.game_pack import _is_link_half, discover_link_pairs
        # Temporary states folder.
        with tempfile.TemporaryDirectory() as tmp:
            # Write empty stand-in files.
            open(os.path.join(tmp, "versus_link_p1.mgba_state"), "wb").close()
            # Player 2 half.
            open(os.path.join(tmp, "versus_link_p2.mgba_state"), "wb").close()
            # Solo file that training may still use.
            open(os.path.join(tmp, "solo.mgba_state"), "wb").close()
            # The pair is found.
            pairs = discover_link_pairs(tmp)
            # One complete pair.
            self.assertEqual(len(pairs), 1)
            # Halves are marked so solo discovery can skip them.
            self.assertTrue(_is_link_half(pairs[0][0]))
            # The solo file is not a link half.
            self.assertFalse(_is_link_half(os.path.join(tmp, "solo.mgba_state")))

    def test_p2_keys_do_not_overlap_p1(self):
        # Both players must be controllable at once.
        import sdl2
        from tools.link_setup import build_p2_key_maps
        from tools.save_mgba_state import build_key_maps
        # Player 1 keys.
        p1, _up1 = build_key_maps()
        # Player 2 keys.
        p2, _up2 = build_p2_key_maps()
        # No shared keycodes.
        self.assertFalse(set(p1) & set(p2))
        # Player 2 still has a punch button.
        self.assertEqual(p2[sdl2.SDLK_u], "a")
        # This machine's build exports the cable API.
        self.assertTrue(link_symbols_available())

    def test_handle_key_targets_the_given_player(self):
        # The handler must not hang a player index off the key dict.
        import sdl2
        from tools.link_setup import _handle_key, build_p2_key_maps

        class _Pad:
            # Record which buttons this fake GBA saw.
            def __init__(self):
                # Presses.
                self.pressed = []

            def press_button(self, name):
                # Remember the button name.
                self.pressed.append(name)

            def release_button(self, name):
                # Releases are unused in this case.
                return None

        class _Link:
            # Two pads, index 0 then 1.
            def __init__(self):
                # Left and right GBAs.
                self.players = [_Pad(), _Pad()]

        # Fake cable.
        link = _Link()
        # Right-hand keymap stays a plain dict.
        p2, _up2 = build_p2_key_maps()
        # U is player 2's A button.
        _handle_key(link, sdl2.SDLK_u, p2, 1, True)
        # Only the right GBA is pressed.
        self.assertEqual(link.players[1].pressed, ["a"])
        # The left GBA stays idle.
        self.assertEqual(link.players[0].pressed, [])


if __name__ == "__main__":
    # Run the self-play unit tests.
    unittest.main()
