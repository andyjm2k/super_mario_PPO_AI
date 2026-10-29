"""Unit tests for the SB3→UI stats writer helpers."""

import unittest
from unittest.mock import MagicMock

from stats_writer import QueueStatsWriter, format_stat_value


class TestStatsWriter(unittest.TestCase):
    """Cover formatting and queue forwarding for logger dumps."""

    def test_format_stat_value_integers(self):
        # FPS should render as a whole number.
        self.assertEqual(format_stat_value("time/fps", 810.4), "810")
        # Episode length mean should render as a whole number.
        self.assertEqual(format_stat_value("rollout/ep_len_mean", 464.2), "464")

    def test_format_stat_value_floats(self):
        # Reward means use compact float formatting.
        self.assertEqual(format_stat_value("rollout/ep_rew_mean", -38.2), "-38.2")
        # Tiny KL values use a compact scientific-ish form.
        formatted = format_stat_value("train/approx_kl", 0.011355076)
        # Ensure the formatter produced a non-empty string.
        self.assertTrue(len(formatted) > 0)

    def test_queue_stats_writer_puts_payload(self):
        # Fake queue capturing put_nowait calls.
        fake_queue = MagicMock()
        # Build the writer against the fake queue.
        writer = QueueStatsWriter(fake_queue)
        # Simulate an SB3 logger.dump payload.
        writer.write(
            {
                "rollout/ep_rew_mean": -38.2,
                "time/fps": 810,
                "train/loss": 9.39,
            },
            key_excluded={},
            step=65536,
        )
        # The writer must enqueue a stats payload.
        fake_queue.put_nowait.assert_called_once()
        # Unpack the payload for assertions.
        payload = fake_queue.put_nowait.call_args[0][0]
        # Payload type must identify stats dumps.
        self.assertEqual(payload["type"], "stats")
        # Step must match the dump step.
        self.assertEqual(payload["step"], 65536)
        # Reward metric must be present.
        self.assertEqual(payload["stats"]["rollout/ep_rew_mean"], -38.2)

    def test_queue_stats_writer_skips_tensorboard_only_keys(self):
        # Fake queue capturing put_nowait calls.
        fake_queue = MagicMock()
        # Build the writer against the fake queue.
        writer = QueueStatsWriter(fake_queue)
        # Exclude a key from stdout the same way SB3 does for some fields.
        writer.write(
            {"time/iterations": 2, "secret/tb_only": 1},
            key_excluded={"secret/tb_only": ("stdout",)},
            step=1,
        )
        # Unpack the payload.
        payload = fake_queue.put_nowait.call_args[0][0]
        # Stdout-visible key should remain.
        self.assertIn("time/iterations", payload["stats"])
        # Stdout-excluded key should be dropped.
        self.assertNotIn("secret/tb_only", payload["stats"])


if __name__ == "__main__":
    # Run the stats-writer unit tests.
    unittest.main()
