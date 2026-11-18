"""Headless check that two linked GBAs finish a Street Fighter VS handshake."""

import os
import unittest

from pdt.emulator.gba_link import GbaLink
from pdt.game_pack import load_game_pack


def _vs_banner(frame) -> bool:
    """True when the top of the frame is the red VS MODE character-select banner."""
    # Avoid a numpy import at collection time if the test is skipped.
    import numpy as np
    # The banner is the top strip of the select screen.
    strip = frame[:12, :, :].astype(np.int16)
    # Red and not green or blue.
    red = (strip[:, :, 0] > 160) & (strip[:, :, 1] < 80) & (strip[:, :, 2] < 80)
    # A real banner covers most of that strip.
    return float(red.mean()) > 0.4


def _tap(link, name: str, hold: int = 8, settle: int = 16) -> None:
    """Press one button on both GBAs, then release it."""
    # Both players receive the same menu input.
    for player in link.players:
        # Hold the named button.
        player.press_button(name)
    # Long enough for the game to see the edge.
    link.tick(hold)
    # Release together.
    for player in link.players:
        # Drop that button.
        player.release_button(name)
    # Let the menu animate.
    link.tick(settle)


class TestLinkHandshake(unittest.TestCase):
    """Boot Street Fighter on two linked cores and enter VS mode together."""

    def test_vs_mode_exchanges_sio_data(self):
        # The cable only exists when this venv has libmgba.
        try:
            # Import probes the real module.
            import mgba.core  # noqa: F401
        except ImportError:
            # Expected until the DLL is installed.
            self.skipTest("mgba is not installed in this venv")
        # The fighting pack names the ROM and BIOS.
        pack = load_game_pack("street_fighter3")
        # Skip when the ROM was not copied into the pack.
        if not os.path.isfile(pack.rom_path) or not os.path.isfile(pack.bios_path or ""):
            # Nothing to boot.
            self.skipTest("Street Fighter ROM or BIOS is missing")
        # Unlimited speed. The window tool is the one that sleeps.
        link = GbaLink(
            rom_path=pack.rom_path,
            bios_path=pack.bios_path,
            skip_bios=True,
            emulation_speed=0,
        )
        try:
            # The C node ids must match the slots or nextEvent updates land on the wrong field.
            self.assertEqual(int(link._nodes[0].id), 0)
            # Player 2 is the slave.
            self.assertEqual(int(link._nodes[1].id), 1)
            # Attract mode is up by this frame. Start before that is ignored.
            link.tick(1400)
            # Attract fades to the title. Wait until PRESS START is live.
            _tap(link, "start", hold=8, settle=90)
            # Title Start opens the mode menu on both GBAs together.
            _tap(link, "start", hold=10, settle=40)
            # Cursor starts on Single. One down is VS.
            _tap(link, "down")
            # Confirm VS on both sides in the same frames.
            _tap(link, "a", hold=12, settle=8)
            # The handshake needs a few seconds of emulated time to reach select.
            link.tick(240)
            # Both screens should be the shared VS character select.
            self.assertTrue(
                _vs_banner(link.players[0].get_rgb_frame()),
                "player 1 never reached the VS character select",
            )
            # Player 2 must be on that same linked select, not still waiting.
            self.assertTrue(
                _vs_banner(link.players[1].get_rgb_frame()),
                "player 2 never reached the VS character select",
            )
        finally:
            # Drop both cores even when an assertion fails.
            link.stop()


if __name__ == "__main__":
    # Run this file directly.
    unittest.main()
