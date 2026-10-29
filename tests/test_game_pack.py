"""Unit tests for game packs, memory banks, and declarative rewards."""

import json
import os
import tempfile
import unittest

from pdt.emulator import BOOT_STATE
from pdt.game_pack import list_game_ids, load_game_pack
from pdt.memory_bank import MemoryBank, parse_address
from pdt.reward_engine import DeclarativeRewardEngine, RewardSpec


class FakeMemory(dict):
    """Dict-backed stand-in for pyboy.memory indexing."""

    def __getitem__(self, item):
        # Missing addresses read as 0 (matches sparse WRAM tests).
        return dict.get(self, item, 0)


class TestGamePack(unittest.TestCase):
    """Verify Mario Deluxe pack discovery and loading."""

    def test_list_includes_mario(self):
        # Mario pack should be installed under games/.
        self.assertIn("mario_deluxe", list_game_ids())

    def test_load_mario_pack(self):
        # Load the migrated Mario pack.
        pack = load_game_pack("mario_deluxe")
        # Display name should be set.
        self.assertIn("Mario", pack.name)
        # ROM path must exist.
        self.assertTrue(os.path.isfile(pack.rom_path))
        # Memory and rewards files must exist.
        self.assertTrue(os.path.isfile(pack.memory_path))
        # Rewards path must exist.
        self.assertTrue(os.path.isfile(pack.rewards_path))
        # At least one start state should be present.
        self.assertGreaterEqual(len(pack.discover_states()), 1)

    def test_street_fighter_boots_until_mgba_state_exists(self):
        # Street Fighter now runs on mGBA, which cannot load old .pba_state files.
        pack = load_game_pack("street_fighter3")
        # Engine id must route through the mGBA backend.
        self.assertEqual(pack.emulator_engine, "mgba")
        # Boot-only mode stays off so a future .mgba_state is preferred.
        self.assertFalse(pack.prefer_boot_start)
        # Training can still start from ROM boot before the user re-records.
        self.assertTrue(pack.allow_boot_start)
        # Discovery ignores versus_start.pba_state.
        states = pack.discover_states()
        # No .mgba_state yet, so the curriculum is the boot sentinel.
        self.assertEqual(states, [BOOT_STATE])

    def test_street_fighter_actions_are_gameplay_only(self):
        # Training must not press pause or menu buttons.
        pack = load_game_pack("street_fighter3")
        # GBA face, shoulder, and d-pad buttons used in a match.
        allowed = {"left", "right", "up", "down", "a", "b", "l", "r"}
        # Walk every discrete action in the pack.
        for action in pack.actions:
            # Each held button must be a gameplay button.
            for button in action.get("buttons", []):
                # Start and Select are excluded from the action set.
                self.assertIn(button, allowed)


class TestMemoryBank(unittest.TestCase):
    """Verify declarative memory decoding."""

    def test_parse_address_hex(self):
        # Hex strings should parse.
        self.assertEqual(parse_address("0xC24B"), 0xC24B)

    def test_tile_percent_decode(self):
        # Load Mario memory bank.
        bank = load_game_pack("mario_deluxe").load_memory()
        # Fake memory with progress tile digit 6 (214) => 50%.
        memory = FakeMemory({0xC24B: 214})
        # Decode LEVEL_PROGRESS.
        self.assertAlmostEqual(bank.read(memory, "LEVEL_PROGRESS"), 50.0)

    def test_merge_import(self):
        # Start from Mario bank.
        bank = load_game_pack("mario_deluxe").load_memory()
        # Build a tiny import bank in a temp file.
        with tempfile.TemporaryDirectory() as tmp:
            # Import path.
            path = os.path.join(tmp, "extra.json")
            # Write one extra signal.
            with open(path, "w", encoding="utf-8") as handle:
                # Minimal memory document.
                json.dump(
                    {
                        "signals": {
                            "CUSTOM_FLAG": {"address": "0xC000", "decode": "raw"}
                        }
                    },
                    handle,
                )
            # Load import.
            imported = MemoryBank.from_file(path)
            # Merge into Mario bank.
            changed = bank.merge_import(imported)
            # One signal should change.
            self.assertEqual(changed, 1)
            # New signal should be readable.
            memory = FakeMemory({0xC000: 7})
            # Decode CUSTOM_FLAG.
            self.assertEqual(bank.read(memory, "CUSTOM_FLAG"), 7.0)


class TestDeclarativeRewardEngine(unittest.TestCase):
    """Verify declarative reward rules for Mario-like signals."""

    def _engine(self):
        # Load Mario pack assets.
        pack = load_game_pack("mario_deluxe")
        # Build engine.
        return DeclarativeRewardEngine(pack.load_rewards(), pack.load_memory())

    def test_living_cost_and_progress(self):
        # Build engine.
        engine = self._engine()
        # Spawn memory: progress 0%, 5 lives, no flag.
        memory = FakeMemory({0xC24B: 208, 0xC17F: 5, 0xC02E: 0, 0xC175: 10})
        # Seed baselines.
        engine.reset(memory)
        # Advance progress to digit 1 (~8.33%).
        memory[0xC24B] = 209
        # Also bump motion so stuck does not fire.
        memory[0xC175] = 11
        # Step the engine.
        reward, terminated, truncated, info = engine.step(memory, episode_length=1)
        # Should earn positive progress reward.
        self.assertGreater(reward, 0.0)
        # Not terminal.
        self.assertFalse(terminated)
        # Not truncated.
        self.assertFalse(truncated)
        # Progress extras present.
        self.assertIn("progress_pct", info)

    def test_death_on_lives_decrement(self):
        # Build engine.
        engine = self._engine()
        # Start with 5 lives.
        memory = FakeMemory({0xC24B: 208, 0xC17F: 5, 0xC02E: 0, 0xC175: 10})
        # Seed.
        engine.reset(memory)
        # Drop a life.
        memory[0xC17F] = 4
        # Step.
        reward, terminated, truncated, info = engine.step(memory, episode_length=5)
        # Death replaces reward with -40.
        self.assertEqual(reward, -40.0)
        # Episode terminates.
        self.assertTrue(terminated)
        # Death component recorded.
        self.assertIn("death", info["reward_components"])

    def test_four_lives_start_is_not_death(self):
        # Build engine.
        engine = self._engine()
        # Start already at 4 lives (historical false-positive case).
        memory = FakeMemory({0xC24B: 208, 0xC17F: 4, 0xC02E: 0, 0xC175: 10})
        # Seed baselines at 4 lives.
        engine.reset(memory)
        # Motion changes so stuck does not fire.
        memory[0xC175] = 11
        # Step without a lives drop.
        reward, terminated, truncated, info = engine.step(memory, episode_length=1)
        # Must not terminate as death.
        self.assertFalse(terminated)
        # Death component must be absent.
        self.assertNotIn("death", info.get("reward_components", {}))

    def test_signal_delta_rewards_decreases(self):
        # Minimal reward spec with one delta rule.
        spec = RewardSpec(
            {
                "clip_min": -100,
                "clip_max": 100,
                "rules": [
                    {
                        "id": "dmg",
                        "type": "signal_delta",
                        "signal": "HP",
                        "decrease_scale": 0.5,
                        "increase_scale": 0.0,
                    }
                ],
            }
        )
        # Tiny memory bank with one raw signal.
        bank = MemoryBank(
            {"signals": {"HP": {"address": "0x10", "decode": "raw"}}}
        )
        # Build engine.
        engine = DeclarativeRewardEngine(spec, bank)
        # Seed at 100 HP.
        memory = FakeMemory({0x10: 100})
        # Reset baselines.
        engine.reset(memory)
        # Drop HP by 10.
        memory[0x10] = 90
        # Step.
        reward, terminated, truncated, info = engine.step(memory, episode_length=1)
        # 10 * 0.5 = 5.0 reward for damage dealt.
        self.assertAlmostEqual(reward, 5.0)
        # Not terminal.
        self.assertFalse(terminated)
        # Load Mario rewards.
        spec = load_game_pack("mario_deluxe").load_rewards()
        # Mutate a coefficient.
        for rule in spec.rules:
            # Find living cost.
            if rule.get("id") == "living_cost":
                # Change value.
                rule["value"] = -0.1
        # Save to a temp file.
        with tempfile.TemporaryDirectory() as tmp:
            # Target path.
            path = os.path.join(tmp, "rewards.json")
            # Persist.
            spec.save(path)
            # Reload.
            reloaded = RewardSpec.from_file(path)
            # Find living cost again.
            living = next(r for r in reloaded.rules if r.get("id") == "living_cost")
            # Value should round-trip.
            self.assertEqual(living["value"], -0.1)


if __name__ == "__main__":
    # Run pack / memory / reward unit tests.
    unittest.main()
