"""Unit tests for MAME P1-vs-P2 learning and snapshot discovery."""

import os
import tempfile
import threading
import time
import unittest

import numpy as np

from pdt.emulator import BOOT_STATE
from pdt.emulator.mame_backend import FakeMameSession, MameBackend
from pdt.emulator.mame_console import MameConsoleSession
from pdt.game_pack import load_game_pack
from pdt.mame_versus import MameVersusEnv, OpponentHook, mirror_health_spec, orient_frame
from env_wrappers import window_for_worker
from training_config import TrainingConfig


class TestMirrorAndOrient(unittest.TestCase):
    """Player 2's reward reads the other health bar, and the view flips."""

    def test_mirror_swaps_health_signals(self):
        # The real 3rd Strike reward document.
        spec = load_game_pack("sfiii3n").load_rewards()
        # Swapped copy.
        mirrored = mirror_health_spec(spec)
        # Find the damage-dealt rule on each document.
        original = next(rule for rule in spec.data["rules"] if rule["id"] == "damage_dealt")
        # The mirrored rule.
        swapped = next(rule for rule in mirrored.data["rules"] if rule["id"] == "damage_dealt")
        # Player 1 is rewarded for player 2 losing health.
        self.assertEqual(original["signal"], "P2_HEALTH")
        # Player 2 is rewarded for player 1 losing health.
        self.assertEqual(swapped["signal"], "P1_HEALTH")

    def test_player_two_view_is_flipped(self):
        # A red pixel on the left.
        frame = np.zeros((4, 6, 3), dtype=np.uint8)
        # Mark it.
        frame[0, 0, 0] = 255
        # Player 1 keeps that pixel on the left.
        self.assertEqual(int(orient_frame(frame, 1)[0, 0, 0]), 255)
        # Player 2's mirror puts it on the right.
        self.assertEqual(int(orient_frame(frame, 2)[0, -1, 0]), 255)


class TestVersusMatch(unittest.TestCase):
    """Both joysticks are driven and each side keeps its own reward."""

    def _env(self, side: int) -> MameVersusEnv:
        # Pack with the real reward rules.
        pack = load_game_pack("sfiii3n")
        # No MAME process.
        session = FakeMameSession()
        # Full bars and a round in progress.
        session.values[0x02068D0A] = 160
        # Opponent.
        session.values[0x020691A2] = 160
        # The fight flag lets a later KO count.
        session.values[0x02011389] = 1
        # Backend over the fake board.
        backend = MameBackend(rom_path=pack.rom_path, session=session, game_id="sfiii3n")
        # Watch the pack addresses.
        backend.bind_memory_bank(pack.load_memory())
        # Versus env pinned to one side and without the coin sequence.
        return MameVersusEnv(
            game_pack=pack,
            backend=backend,
            learner_side=side,
            boot_actions=[],
            state_paths=[BOOT_STATE],
        )

    def test_both_players_act_and_rewards_differ(self):
        # Player 1 is the stored trajectory.
        env = self._env(1)
        # Bind player 2 to a jab.
        env.set_opponent(lambda _obs: 9)
        # Baselines at full health.
        env.reset()
        # Player 2 takes 10 damage.
        env.emu._session.values[0x020691A2] = 150
        # Player 1 walks forward. Action 2 is right.
        _obs, reward, terminated, _trunc, info = env.step(2)
        # 10 health times 0.35, minus the living cost.
        self.assertAlmostEqual(reward, 3.5 - 0.01, places=3)
        # The round is not over.
        self.assertFalse(terminated)
        # Player 2's own return is the damage they took.
        self.assertLess(info["reward_p2"], 0.0)
        # Both joysticks were sent.
        actions = env.emu._session.actions
        # Player 1 walks.
        self.assertIn((":INPUTS", "P1 Right"), actions)
        # Player 2 jabs.
        self.assertIn((":INPUTS", "P2 Jab Punch"), actions)
        # The hook can rebind the policy.
        hook = OpponentHook(env)
        # A saved-model path is what the parallel workers receive.
        hook.set_opponent(os.path.join(tempfile.gettempdir(), "missing_opponent.zip"))
        # It landed on the versus env as a path, not a callable.
        self.assertTrue(env._opponent_path.endswith("missing_opponent.zip"))
        # Stop the fake.
        env.close()

    def test_player_two_learner_is_punished_for_taking_damage(self):
        # This episode stores player 2.
        env = self._env(2)
        # Player 1 will jab.
        env.set_opponent(lambda _obs: 9)
        # Full bars.
        env.reset()
        # Player 2 loses 10 health.
        env.emu._session.values[0x020691A2] = 150
        # Player 2's action is index 2, walk.
        _obs, reward, _terminated, _trunc, info = env.step(2)
        # Damage taken is negative for the fighter who was hit.
        self.assertLess(reward, -1.0)
        # The info still reports which side was stored.
        self.assertEqual(info["learner_side"], 2)
        # Player 2 walked, so their right button was held.
        self.assertIn((":INPUTS", "P2 Right"), env.emu._session.actions)
        # Close.
        env.close()


class TestVisibleWorkers(unittest.TestCase):
    """Only the UI's visual-agent count opens a MAME window."""

    def test_later_workers_are_headless(self):
        # Four panes, sixteen workers, screens on.
        config = TrainingConfig(n_envs=16, visual_agents=4, show_visuals=True)
        # The first four are visible.
        self.assertEqual(window_for_worker(config, 0), "SDL2")
        self.assertEqual(window_for_worker(config, 3), "SDL2")
        # The rest do not open a window.
        self.assertEqual(window_for_worker(config, 4), "null")
        self.assertEqual(window_for_worker(config, 15), "null")
        # The toggle hides every window.
        config.show_visuals = False
        self.assertEqual(window_for_worker(config, 0), "null")
    """A saved .sta is a training start, and the command channel can answer."""

    def test_pack_accepts_sta_files(self):
        # The 3rd Strike pack.
        pack = load_game_pack("sfiii3n")
        # Snapshots are not ignored.
        self.assertFalse(pack.prefer_boot_start)
        # The saver's extension is listed.
        self.assertIn(".sta", pack.meta["state_extensions"])

    def test_run_command_waits_for_the_script(self):
        # No MAME process.
        session = MameConsoleSession("mame", ".", "sfiii3n", [(1, 1)], False, start=False)
        # The result file the script would write.
        result = os.path.join(session._dir, "result.txt")

        def answer():
            # Let the waiter start.
            time.sleep(0.05)
            # Successful save.
            with open(result, "w", encoding="ascii") as handle:
                # The status the waiter expects.
                handle.write("ok")

        # Answer from another thread, like the frame callback.
        threading.Thread(target=answer, daemon=True).start()
        # A path that does not need to exist for this fake.
        session.run_command("save", os.path.join(tempfile.gettempdir(), "versus.sta"), timeout=2)
        # The session directory is still there until close.
        session.close()
        # Removed.
        self.assertFalse(os.path.isdir(session._dir))


if __name__ == "__main__":
    # Run the versus tests.
    unittest.main()
