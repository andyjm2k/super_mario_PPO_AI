"""Unit tests for memory address helpers."""

import unittest

from memory_map import (
    MarioMemoryMap,
    digits_to_int,
    is_flag_reached,
    is_game_timer_critical,
    is_life_lost,
    progress_percent_from_tile,
    read_score_digits,
    tile_to_digit,
)


class FakeMemory(dict):
    """Dict-backed stand-in for pyboy.memory indexing."""

    def __getitem__(self, item):
        # Delegate to the underlying dict mapping.
        return dict.__getitem__(self, item)


class TestMarioMemoryMap(unittest.TestCase):
    """Verify tile decoding and address helpers."""

    def test_tile_to_digit_known_values(self):
        # Digit 0 uses tile 208.
        self.assertEqual(tile_to_digit(208), 0)
        # Digit 5 uses tile 213.
        self.assertEqual(tile_to_digit(213), 5)
        # Digit 12 uses tile 220.
        self.assertEqual(tile_to_digit(220), 12)

    def test_tile_to_digit_out_of_range(self):
        # Values below the HUD digit range become 0.
        self.assertEqual(tile_to_digit(0), 0)
        # Values above the HUD digit range become 0.
        self.assertEqual(tile_to_digit(255), 0)

    def test_digits_to_int_leading_zeros(self):
        # Leading zeros should be ignored when assembling the score.
        self.assertEqual(digits_to_int([0, 0, 1, 2, 3]), 123)

    def test_digits_to_int_all_zero(self):
        # An all-zero digit list should yield score 0.
        self.assertEqual(digits_to_int([0, 0, 0, 0, 0]), 0)

    def test_read_score_digits(self):
        # Build a fake memory map with known score tiles.
        memory = FakeMemory(
            {
                MarioMemoryMap.SCORE_DIGIT_0: 208,
                MarioMemoryMap.SCORE_DIGIT_1: 209,
                MarioMemoryMap.SCORE_DIGIT_2: 210,
                MarioMemoryMap.SCORE_DIGIT_3: 211,
                MarioMemoryMap.SCORE_DIGIT_4: 212,
            }
        )
        # Digits should decode to 0,1,2,3,4.
        self.assertEqual(read_score_digits(memory), [0, 1, 2, 3, 4])

    def test_progress_percent_from_tile(self):
        # Half-stage progress tile 214 (digit 6) is 50 percent.
        self.assertAlmostEqual(progress_percent_from_tile(214), 50.0)

    def test_is_life_lost_detects_decrement(self):
        # Lives counter dropping from 5 to 4 means a life was lost.
        memory = FakeMemory({MarioMemoryMap.LIVES_REMAINING: 4})
        # Helper should report True when previous lives were higher.
        self.assertTrue(is_life_lost(memory, previous_lives=5))

    def test_is_life_lost_ignores_stable_four_lives(self):
        # Starting an episode with 4 lives left is not a death.
        memory = FakeMemory({MarioMemoryMap.LIVES_REMAINING: 4})
        # Without a higher baseline, value 4 must not terminate the episode.
        self.assertFalse(is_life_lost(memory, previous_lives=4))
        # Missing baseline also must not false-trigger.
        self.assertFalse(is_life_lost(memory))

    def test_is_flag_reached(self):
        # Flag-state value 80 means the flag was reached.
        memory = FakeMemory({MarioMemoryMap.FLAG_STATE: 80})
        # Helper should report True.
        self.assertTrue(is_flag_reached(memory))

    def test_is_game_timer_critical(self):
        # Timer pattern 001 matches the critical timer heuristic.
        memory = FakeMemory(
            {
                MarioMemoryMap.TIMER_DIGIT_0: 208,
                MarioMemoryMap.TIMER_DIGIT_1: 208,
                MarioMemoryMap.TIMER_DIGIT_2: 209,
            }
        )
        # Helper should report True.
        self.assertTrue(is_game_timer_critical(memory))


if __name__ == "__main__":
    # Run the memory-map unit tests.
    unittest.main()
