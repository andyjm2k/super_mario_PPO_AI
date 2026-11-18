"""Unit tests for the MAME Street Fighter III backend and reward map."""

import os
import unittest

import numpy as np

from pdt.emulator.factory import normalize_engine_name
from pdt.emulator.mame_backend import (
    FakeMameSession,
    MameBackend,
    MameMemoryView,
    find_mame_binary,
)
from pdt.emulator.mame_pads import pad_for
from pdt.game_pack import load_game_pack
from pdt.memory_bank import MemoryBank
from pdt.reward_engine import DeclarativeRewardEngine


class TestMameEngineName(unittest.TestCase):
    """The factory must accept the arcade id."""

    def test_finds_project_or_env_binary(self):
        # The project copy or PDT_MAME_BINARY must resolve to a real exe.
        found = find_mame_binary()
        # Installed next to the repo.
        self.assertIsNotNone(found)
        # The file is on disk.
        self.assertTrue(os.path.isfile(found))

    def test_mame_alias(self):
        # Canonical name.
        self.assertEqual(normalize_engine_name("mame"), "mame")
        # Toolkit alias.
        self.assertEqual(normalize_engine_name("MAMEToolkit"), "mame")


class TestS16Health(unittest.TestCase):
    """Arcade health is a signed 16-bit value, not one byte."""

    def test_s16_roundtrip(self):
        # Two-byte health bar.
        view = MameMemoryView()
        # Full bar used by 3rd Strike.
        view.write_unit(0x02068D0A, 160, 2)
        # Pack signal pointing at that address.
        bank = MemoryBank(
            {
                "signals": {
                    "P1_HEALTH": {"address": "0x02068D0A", "decode": "s16"}
                }
            }
        )
        # The reward engine must see 160, not just the low byte.
        self.assertEqual(bank.read(view, "P1_HEALTH"), 160.0)


class TestMameBackendFake(unittest.TestCase):
    """Step a fake MAME session through the pack reward wrapper."""

    def test_damage_dealt_uses_p2_health(self):
        # Load the arcade pack. The rom folder is created if missing.
        pack = load_game_pack("sfiii3n")
        # The NO CD driver is the set MAME accepts for this folder.
        self.assertEqual(pack.meta.get("mame_game"), "sfiii3n")
        # Scripted board, no mame.exe.
        session = FakeMameSession()
        # Both fighters start at full health.
        session.values[0x02068D0A] = 160
        # Opponent health.
        session.values[0x020691A2] = 160
        # Backend under test.
        backend = MameBackend(rom_path=pack.rom_path, session=session, game_id="sfiii3n")
        # Watch the pack addresses.
        backend.bind_memory_bank(pack.load_memory())
        # One step publishes RAM.
        backend.tick(1)
        # Reward engine from the pack.
        engine = DeclarativeRewardEngine(pack.load_rewards(), pack.load_memory())
        # Seed baselines at full health.
        engine.reset(backend.memory)
        # Opponent takes 10 damage.
        session.values[0x020691A2] = 150
        # Publish the new bar.
        backend.tick(1)
        # Shape one step.
        reward, _terminated, _truncated, _info = engine.step(backend.memory, episode_length=40)
        # 10 health * 0.35, minus the living cost.
        self.assertAlmostEqual(reward, 3.5 - 0.01, places=4)
        # Close the fake.
        backend.stop()
        # The session was closed.
        self.assertTrue(session.closed)

    def test_player2_gets_the_lagged_buttons(self):
        # Fake board.
        session = FakeMameSession()
        # Backend.
        backend = MameBackend(rom_path=".", session=session)
        # Player 1 walks forward and jabs.
        backend.press_player(1, "right")
        backend.press_player(1, "lp")
        # Player 2 blocks.
        backend.press_player(2, "left")
        # One step.
        backend.tick(1)
        # Both joysticks were sent.
        self.assertIn((":INPUTS", "P1 Right"), session.actions)
        # Jab punch.
        self.assertIn((":INPUTS", "P1 Jab Punch"), session.actions)
        # Player 2 left.
        self.assertIn((":INPUTS", "P2 Left"), session.actions)
        # Frame was stored.
        self.assertEqual(backend.get_rgb_frame().shape[2], 3)
        # Shut down.
        backend.stop()


class TestSfa3Pack(unittest.TestCase):
    """Alpha 3 uses CPS2 ports and a round clock around the KO rules."""

    def test_pad_uses_cps2_buttons(self):
        # Alpha 3 light punch is button 1 on IN0.
        self.assertEqual(pad_for("sfa3")[1]["lp"], (":IN0", "P1 Button 1"))
        # Heavy kick for player 2 sits on IN2.
        self.assertEqual(pad_for("SFA3")[2]["hk"], (":IN2", "P2 Button 6"))
        # 3rd Strike keeps the CPS3 names, including the current Start labels.
        self.assertEqual(pad_for("sfiii3n")[1]["lp"], (":INPUTS", "P1 Jab Punch"))
        self.assertEqual(pad_for("sfiii3n")[1]["start"], (":INPUTS", "P1 Start"))
        self.assertEqual(pad_for("sfiii3n")[2]["start"], (":INPUTS", "P2 Start"))

    def test_health_word_and_ko_needs_the_clock(self):
        # Load the Alpha 3 pack.
        pack = load_game_pack("sfa3")
        # Driver the zip is named for.
        self.assertEqual(pack.meta.get("mame_game"), "sfa3")
        # Scripted board.
        session = FakeMameSession()
        # Full bars and a running 90-second clock.
        session.values[0xFF8450] = 144
        session.values[0xFF8850] = 144
        session.values[0xFF8109] = 90
        # Backend bound to this driver.
        backend = MameBackend(rom_path=pack.rom_path, session=session, game_id="sfa3")
        # Watch the pack addresses.
        backend.bind_memory_bank(pack.load_memory())
        # Publish the full bars.
        backend.tick(1)
        # Reward engine from the pack.
        engine = DeclarativeRewardEngine(pack.load_rewards(), pack.load_memory())
        # Seed baselines.
        engine.reset(backend.memory)
        # Opponent takes 10 damage.
        session.values[0xFF8850] = 134
        # Publish it.
        backend.tick(1)
        # Shape one step. Too early for a KO.
        reward, terminated, _truncated, _info = engine.step(backend.memory, episode_length=40)
        # 10 * 0.35, minus living cost. Not a KO yet.
        self.assertAlmostEqual(reward, 3.5 - 0.01, places=4)
        # Round is still going.
        self.assertFalse(terminated)
        # Title screen: both bars and the clock are zero.
        session.values[0xFF8450] = 0
        session.values[0xFF8850] = 0
        session.values[0xFF8109] = 0
        # Publish the menu.
        backend.tick(1)
        # Past the grace window.
        _reward, terminated, _truncated, _info = engine.step(backend.memory, episode_length=40)
        # A zero bar with no clock is not a knockout.
        self.assertFalse(terminated)
        # Clock starts, then player 2 is KOed.
        session.values[0xFF8109] = 80
        session.values[0xFF8450] = 144
        session.values[0xFF8850] = 0
        # Publish the KO frame.
        backend.tick(1)
        # Win terminal.
        _reward, terminated, _truncated, _info = engine.step(backend.memory, episode_length=40)
        # The clock gate is open and P2 health is 0.
        self.assertTrue(terminated)
        # Player 1 jab must use the CPS2 field.
        backend.press_player(1, "lp")
        backend.tick(1)
        # Button 1, not the 3rd Strike punch name.
        self.assertIn((":IN0", "P1 Button 1"), session.actions)
        # Shut down.
        backend.stop()


if __name__ == "__main__":
    # Run the MAME unit tests.
    unittest.main()
