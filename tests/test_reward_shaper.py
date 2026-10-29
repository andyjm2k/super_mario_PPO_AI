"""Unit tests for the progress-first Mario reward shaper."""

import unittest

from reward_shaper import MarioRewardShaper, RewardConfig


class TestMarioRewardShaper(unittest.TestCase):
    """Cover the main shaping paths used during training."""

    def setUp(self):
        # Fresh shaper with default coefficients before each test.
        self.shaper = MarioRewardShaper(RewardConfig())
        # Reset episode baselines to a known spawn state.
        self.shaper.reset(progress_pct=0.0, score=0)

    def test_reset_clears_trackers(self):
        # Mutate trackers as if an episode already ran.
        self.shaper.best_progress_pct = 40.0
        # Reset should restore the provided baselines.
        self.shaper.reset(progress_pct=10.0, score=50)
        # Previous progress must match the reset argument.
        self.assertEqual(self.shaper.prev_progress_pct, 10.0)
        # Previous score must match the reset argument.
        self.assertEqual(self.shaper.prev_score, 50)
        # Best progress must match the reset argument.
        self.assertEqual(self.shaper.best_progress_pct, 10.0)
        # Stuck counter must be cleared.
        self.assertEqual(self.shaper.stuck_steps, 0)

    def test_progress_reward_positive(self):
        # Moving into new territory should produce a positive shaped reward.
        reward, terminated, truncated, info = self.shaper.shape(
            progress_pct=5.0,
            score=0,
            flag_reached=False,
            life_lost=False,
            timed_out=False,
            is_stuck=False,
            episode_length=10,
        )
        # Reward should exceed the living cost alone.
        self.assertGreater(reward, 0.0)
        # Episode should continue.
        self.assertFalse(terminated)
        # Episode should not truncate.
        self.assertFalse(truncated)
        # Progress component should be present.
        self.assertIn("progress", info["reward_components"])
        # New-best bonus should fire with the first advance.
        self.assertIn("new_best", info["reward_components"])
        # No-forward penalty must not apply when claiming new territory.
        self.assertNotIn("no_forward", info["reward_components"])

    def test_recrossing_old_progress_is_not_rewarded(self):
        # Advance to mid-level once.
        self.shaper.shape(
            progress_pct=20.0,
            score=0,
            flag_reached=False,
            life_lost=False,
            timed_out=False,
            is_stuck=False,
            episode_length=10,
        )
        # Retreat toward the start.
        self.shaper.shape(
            progress_pct=10.0,
            score=0,
            flag_reached=False,
            life_lost=False,
            timed_out=False,
            is_stuck=False,
            episode_length=11,
        )
        # Re-advance over already-visited ground must not pay progress.
        reward, _, _, info = self.shaper.shape(
            progress_pct=15.0,
            score=0,
            flag_reached=False,
            life_lost=False,
            timed_out=False,
            is_stuck=False,
            episode_length=12,
        )
        # Dense progress must not fire for re-crossing.
        self.assertNotIn("progress", info["reward_components"])
        # Stall / re-cross penalty should apply instead.
        self.assertIn("no_forward", info["reward_components"])
        # Net reward for re-crossing should be negative.
        self.assertLess(reward, 0.0)

    def test_back_and_forth_is_net_negative(self):
        # Claim some new ground.
        first, _, _, _ = self.shaper.shape(
            progress_pct=10.0,
            score=0,
            flag_reached=False,
            life_lost=False,
            timed_out=False,
            is_stuck=False,
            episode_length=10,
        )
        # Immediately retreat.
        second, _, _, info = self.shaper.shape(
            progress_pct=5.0,
            score=0,
            flag_reached=False,
            life_lost=False,
            timed_out=False,
            is_stuck=False,
            episode_length=11,
        )
        # Backward component must be present on the retreat.
        self.assertIn("backward", info["reward_components"])
        # The retreat step itself must be negative.
        self.assertLess(second, 0.0)
        # Oscillating after a gain must not be a profitable loop overall
        # once the agent tries to re-earn the same stretch.
        third, _, _, third_info = self.shaper.shape(
            progress_pct=10.0,
            score=0,
            flag_reached=False,
            life_lost=False,
            timed_out=False,
            is_stuck=False,
            episode_length=12,
        )
        # Re-crossing to 10% (already best) must not pay progress again.
        self.assertNotIn("progress", third_info["reward_components"])
        # Combined retreat + re-cross must erase the earlier gain incentive.
        self.assertLess(second + third, 0.0)
        # Keep first reward unused beyond ensuring it was positive.
        self.assertGreater(first, 0.0)

    def test_no_forward_penalty_when_stalling(self):
        # Standing still must be penalised.
        reward, _, _, info = self.shaper.shape(
            progress_pct=0.0,
            score=9999,
            flag_reached=False,
            life_lost=False,
            timed_out=False,
            is_stuck=False,
            episode_length=10,
        )
        # No-forward component must be recorded.
        self.assertIn("no_forward", info["reward_components"])
        # Score must never appear in reward components.
        self.assertNotIn("score", info["reward_components"])
        # Net reward should be negative.
        self.assertLess(reward, 0.0)

    def test_score_never_rewarded(self):
        # Even with progress, score deltas must not add reward.
        _, _, _, info = self.shaper.shape(
            progress_pct=5.0,
            score=10_000,
            flag_reached=False,
            life_lost=False,
            timed_out=False,
            is_stuck=False,
            episode_length=10,
        )
        # No score component should exist.
        self.assertNotIn("score", info["reward_components"])
        # Absolute score may still be reported in info for diagnostics.
        self.assertEqual(info["score"], 10_000)

    def test_backward_penalty(self):
        # Seed progress at mid-level.
        self.shaper.reset(progress_pct=20.0, score=0)
        # Moving backward should apply the backward penalty component.
        reward, _, _, info = self.shaper.shape(
            progress_pct=15.0,
            score=0,
            flag_reached=False,
            life_lost=False,
            timed_out=False,
            is_stuck=False,
            episode_length=10,
        )
        # Backward component must be recorded.
        self.assertIn("backward", info["reward_components"])
        # No-forward penalty should also apply while retreating.
        self.assertIn("no_forward", info["reward_components"])
        # Net reward should be negative.
        self.assertLess(reward, 0.0)

    def test_flag_terminates(self):
        # Reaching the flag should terminate with a large bonus.
        reward, terminated, truncated, info = self.shaper.shape(
            progress_pct=100.0,
            score=0,
            flag_reached=True,
            life_lost=False,
            timed_out=False,
            is_stuck=False,
            episode_length=10,
        )
        # Flag component should be present.
        self.assertIn("flag", info["reward_components"])
        # Episode should terminate successfully.
        self.assertTrue(terminated)
        # Truncation is not required for flag completion.
        self.assertFalse(truncated)
        # Reward should be strongly positive.
        self.assertGreater(reward, 10.0)

    def test_death_terminates(self):
        # Losing a life should terminate with a penalty.
        reward, terminated, truncated, info = self.shaper.shape(
            progress_pct=10.0,
            score=0,
            flag_reached=False,
            life_lost=True,
            timed_out=False,
            is_stuck=False,
            episode_length=10,
        )
        # Death component should be present.
        self.assertIn("death", info["reward_components"])
        # Episode should terminate.
        self.assertTrue(terminated)
        # Reward should be negative.
        self.assertLess(reward, 0.0)

    def test_stuck_truncates_after_limit(self):
        # Force the stuck counter past the truncation limit.
        for step in range(105, 906):
            # Keep reporting stuck after the grace period.
            _, terminated, truncated, info = self.shaper.shape(
                progress_pct=0.0,
                score=0,
                flag_reached=False,
                life_lost=False,
                timed_out=False,
                is_stuck=True,
                episode_length=step,
                stuck_grace=100,
                stuck_limit=800,
            )
            # Stop once truncation triggers.
            if truncated:
                # Truncation should happen without a true termination.
                self.assertFalse(terminated)
                # Stuck steps should be reported in info.
                self.assertGreaterEqual(info["stuck_steps"], 800)
                # Exit the test successfully.
                return
        # Fail if truncation never triggered.
        self.fail("Stuck truncation never triggered")


if __name__ == "__main__":
    # Run the reward-shaper unit tests.
    unittest.main()
