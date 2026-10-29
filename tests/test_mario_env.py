"""Unit tests for the generic PyBoy env helpers with a mocked emulator."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np


class _FakeBackend:
    """Minimal emulator backend stand-in for unit tests."""

    def __init__(self):
        # Dict-like memory map covering Mario WRAM/VRAM ranges.
        self.memory = {addr: 0 for addr in range(0x9800, 0xC300)}
        # Seed lives baseline.
        self.memory[0xC17F] = 5
        # Track button releases for assertions.
        self.released = []

    def tick(self, count=1):
        # No-op frame advance.
        return None

    def press_button(self, name):
        # No-op press.
        return None

    def release_button(self, name):
        # Record releases.
        self.released.append(name)

    def release_all(self):
        # Mimic release of the common set.
        for name in ("left", "right", "up", "down", "a", "b", "start", "select"):
            # Record each release.
            self.released.append(name)

    def get_rgb_frame(self):
        # GB-sized RGB frame.
        return np.zeros((144, 160, 3), dtype=np.uint8)

    def set_emulation_speed(self, speed):
        # No-op speed setter.
        return None

    def load_state_file(self, path):
        # No-op state load.
        return None

    def stop(self):
        # No-op stop.
        return None


class TestPyBoyEnvHelpers(unittest.TestCase):
    """Test env helpers without launching a real emulator window."""

    def _build_env(self):
        # Fake backend instance shared with assertions.
        fake = _FakeBackend()
        # Patch the factory so constructing the env never boots PyBoy.
        with patch("pdt.pyboy_env.create_emulator_backend", return_value=fake):
            # Import inside the patch.
            from pdt.pyboy_env import PyBoyEnv

            # Build the env with an explicit fake state list to skip discovery.
            with patch.object(PyBoyEnv, "_resolve_states", return_value=["fake.state"]):
                # Construct the Mario pack environment under test.
                env = PyBoyEnv(game_id="mario_deluxe", window="null")
        # Return both the env and the fake backend.
        return env, fake

    def test_observation_space_shape(self):
        # Build a mocked environment.
        env, _ = self._build_env()
        # Env emits a single grayscale frame; wrappers stack later.
        self.assertEqual(env.observation_space.shape, (64, 80, 1))
        # Close to stop any background resources.
        env.close()

    def test_get_observation_shape(self):
        # Build a mocked environment.
        env, _ = self._build_env()
        # Force a real RGB ndarray through the preprocess path.
        env.get_rgb_frame = lambda: np.zeros((144, 160, 3), dtype=np.uint8)
        # Observation must be cropped/resized to the configured size.
        obs = env.get_observation()
        # Confirm the single-frame observation shape.
        self.assertEqual(obs.shape, (64, 80, 1))
        # Close the env.
        env.close()

    def test_action_space_size(self):
        # Build a mocked environment.
        env, _ = self._build_env()
        # Mario pack defines six discrete actions.
        self.assertEqual(env.action_space.n, 6)
        # Close the env.
        env.close()

    def test_release_all_keys_calls_backend(self):
        # Build a mocked environment.
        env, fake = self._build_env()
        # Release every movement/jump key.
        env.release_all_keys()
        # Left should be released.
        self.assertIn("left", fake.released)
        # Right should be released.
        self.assertIn("right", fake.released)
        # Jump should be released.
        self.assertIn("a", fake.released)
        # Close the env.
        env.close()

    def test_score_signal_via_memory_bank(self):
        # Build a mocked environment.
        env, fake = self._build_env()
        # Write score tiles for 01234 -> 1234.
        fake.memory[0xC103] = 208
        # Digit 1.
        fake.memory[0xC104] = 209
        # Digit 2.
        fake.memory[0xC105] = 210
        # Digit 3.
        fake.memory[0xC106] = 211
        # Digit 4.
        fake.memory[0xC107] = 212
        # Declarative SCORE signal should assemble correctly.
        self.assertEqual(env.memory_bank.read(fake.memory, "SCORE"), 1234.0)
        # Close the env.
        env.close()

    def test_engine_is_pyboy_for_mario(self):
        # Build a mocked environment.
        env, _ = self._build_env()
        # Mario pack must select the PyBoy backend.
        self.assertEqual(env.engine, "pyboy")
        # Close the env.
        env.close()


if __name__ == "__main__":
    # Run the environment helper unit tests.
    unittest.main()
