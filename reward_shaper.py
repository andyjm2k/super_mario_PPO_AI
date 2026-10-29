"""Legacy Mario reward shaper (unit-test compatible).

Production training uses ``pdt.reward_engine.DeclarativeRewardEngine`` driven by
``games/<id>/rewards.json``. This module remains for older tests and scripts.
"""

import numpy as np


class RewardConfig:
    """Tunable coefficients for the Mario reward function."""

    def __init__(self):
        # Small constant cost each step so lingering is expensive.
        self.living_cost = -0.05
        # Reward only for progress beyond the episode furthest point.
        self.progress_scale = 12.0
        # Extra bonus when Mario sets a new episode-best progress.
        self.new_best_progress_bonus = 4.0
        # Strong penalty for moving backward (kills left-right farming).
        self.backward_penalty = -4.0
        # Penalty when not setting new furthest progress (stall / re-cross).
        self.no_forward_penalty = -0.2
        # Extra penalty when oscillating without new territory.
        self.oscillation_penalty = -0.5
        # How many recent direction flips count as oscillation farming.
        self.oscillation_window = 6
        # Large terminal bonus for completing a level.
        self.flag_bonus = 200.0
        # Terminal penalty for falling / losing a life.
        self.death_penalty = -40.0
        # Terminal penalty for episode timeout.
        self.timeout_penalty = -20.0
        # Extra per-step penalty while Mario is stuck in place.
        self.stuck_penalty = -0.1
        # Clip bounds keep the critic numerically stable.
        self.clip_min = -50.0
        # Upper clip bound for a single step reward.
        self.clip_max = 50.0


class MarioRewardShaper:
    """Compute dense rewards only for new furthest level progress."""

    def __init__(self, config=None):
        # Allow callers to inject custom coefficients or use defaults.
        self.config = config if config is not None else RewardConfig()
        # Track previous progress percent for direction / delta checks.
        self.prev_progress_pct = 0.0
        # Track previous score only for info logging (never rewarded).
        self.prev_score = 0
        # Track the furthest progress reached this episode.
        self.best_progress_pct = 0.0
        # Count consecutive stuck steps for truncation decisions.
        self.stuck_steps = 0
        # Recent progress direction signs (+1 forward, -1 back, 0 flat).
        self.recent_directions = []

    def reset(self, progress_pct=0.0, score=0):
        """Reset episode-local tracking after an environment reset."""
        # Seed previous progress from the start-state baseline.
        self.prev_progress_pct = float(progress_pct)
        # Keep score baseline for diagnostics only.
        self.prev_score = int(score)
        # Episode-best progress starts at the spawn progress.
        self.best_progress_pct = float(progress_pct)
        # Clear stuck counter at the beginning of every episode.
        self.stuck_steps = 0
        # Clear oscillation history for the new episode.
        self.recent_directions = []

    def shape(
        self,
        progress_pct,
        score,
        flag_reached,
        life_lost,
        timed_out,
        is_stuck,
        episode_length,
        stuck_grace=100,
        stuck_limit=800,
    ):
        """Return (reward, terminated, truncated, info) for one env step."""
        # Start each step with a small living cost.
        reward = self.config.living_cost
        # Dictionary of reward components for logging / UI.
        components = {}
        # Current progress as a float for comparisons.
        progress_pct = float(progress_pct)
        # Step delta vs the previous reading (can be re-crossing old ground).
        delta_progress = progress_pct - self.prev_progress_pct
        # How far beyond the episode furthest point Mario reached.
        new_territory = progress_pct - self.best_progress_pct

        # Pay dense progress only when Mario exceeds episode-best distance.
        if new_territory > 0.0:
            # Scale only the newly claimed percent of the stage.
            progress_reward = new_territory * self.config.progress_scale
            # Add the dense progress signal.
            reward += progress_reward
            # Record the progress component.
            components["progress"] = progress_reward
            # Flat bonus for pushing the frontier further.
            reward += self.config.new_best_progress_bonus
            # Record the new-best component.
            components["new_best"] = self.config.new_best_progress_bonus
            # Update the episode frontier.
            self.best_progress_pct = progress_pct
            # Clear oscillation history after genuine new progress.
            self.recent_directions = []
        elif delta_progress < 0.0:
            # Punish retreating so re-crossing the same stretch cannot pay.
            reward += self.config.backward_penalty
            # Record the backward component.
            components["backward"] = self.config.backward_penalty
            # Also apply the no-forward stall penalty while retreating.
            reward += self.config.no_forward_penalty
            # Record the stall component.
            components["no_forward"] = self.config.no_forward_penalty
        else:
            # No new territory: standing still OR re-entering visited ground.
            reward += self.config.no_forward_penalty
            # Record the stall / re-cross component.
            components["no_forward"] = self.config.no_forward_penalty

        # Track recent movement direction for oscillation detection.
        if delta_progress > 0.0:
            # Forward relative to last step.
            direction = 1
        elif delta_progress < 0.0:
            # Backward relative to last step.
            direction = -1
        else:
            # Flat progress reading.
            direction = 0
        # Append non-zero directions into the oscillation window.
        if direction != 0:
            # Keep a short history of signed moves.
            self.recent_directions.append(direction)
            # Trim to the configured window length.
            window = self.config.oscillation_window
            # Drop older signs outside the window.
            self.recent_directions = self.recent_directions[-window:]
        # Penalise rapid left/right flipping without new territory.
        if (
            new_territory <= 0.0
            and len(self.recent_directions) >= self.config.oscillation_window
            and self._is_oscillating(self.recent_directions)
        ):
            # Apply an extra anti-wiggle penalty.
            reward += self.config.oscillation_penalty
            # Record the oscillation component.
            components["oscillation"] = self.config.oscillation_penalty

        # Remember progress for the next delta calculation.
        self.prev_progress_pct = progress_pct
        # Track score for info only — never add score to the reward.
        self.prev_score = int(score)

        # Default termination flags before terminal checks.
        terminated = False
        # Truncation covers timeouts / stuck cuts, not true task failure.
        truncated = False
        # Apply stuck logic only after a short movement grace period.
        if episode_length > stuck_grace and is_stuck:
            # Increment the consecutive stuck counter.
            self.stuck_steps += 1
            # Apply an extra stuck penalty on top of no-forward.
            reward += self.config.stuck_penalty
            # Record the stuck component.
            components["stuck"] = self.config.stuck_penalty
            # Truncate if Mario has been idle for too long.
            if self.stuck_steps >= stuck_limit:
                # Mark the episode as truncated due to being stuck.
                truncated = True
        else:
            # Reset stuck counter whenever Mario is moving again.
            self.stuck_steps = 0

        # Completing the level is a strong positive terminal signal.
        if flag_reached:
            # Add the flag completion bonus on top of any progress earned.
            reward += self.config.flag_bonus
            # Record the flag component.
            components["flag"] = self.config.flag_bonus
            # Level complete ends the episode successfully.
            terminated = True
        # Falling / losing a life must dominate other shaping this step.
        if life_lost:
            # Replace the step reward so death cannot look profitable.
            reward = self.config.death_penalty
            # Record the death component.
            components["death"] = self.config.death_penalty
            # Death terminates the episode.
            terminated = True
        # Timeouts are truncated with a milder penalty.
        if timed_out and not terminated:
            # Add the timeout penalty only when death/flag did not already end it.
            reward += self.config.timeout_penalty
            # Record the timeout component.
            components["timeout"] = self.config.timeout_penalty
            # Timeout truncates rather than terminating success/failure.
            truncated = True

        # Clip the final reward to keep value estimates stable.
        reward = float(np.clip(reward, self.config.clip_min, self.config.clip_max))
        # Package diagnostic info for callbacks and the training UI.
        info = {
            "reward_components": components,
            "progress_pct": progress_pct,
            "best_progress_pct": float(self.best_progress_pct),
            "score": int(score),
            "stuck_steps": int(self.stuck_steps),
        }
        # Return the shaped reward and episode end flags.
        return reward, terminated, truncated, info

    def _is_oscillating(self, directions):
        """Return True when recent signed moves flip direction repeatedly."""
        # Need at least two signs to detect a flip.
        if len(directions) < 2:
            # Not enough history.
            return False
        # Count how many times consecutive signs disagree.
        flips = 0
        # Walk adjacent pairs in the window.
        for index in range(1, len(directions)):
            # A flip is a sign change between successive moves.
            if directions[index] != directions[index - 1]:
                # Count this direction change.
                flips += 1
        # Treat many flips inside the short window as oscillation farming.
        return flips >= (len(directions) // 2)
