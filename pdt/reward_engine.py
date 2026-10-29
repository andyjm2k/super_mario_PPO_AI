"""Declarative reward engine driven by a game pack rewards.json."""

from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pdt.memory_bank import MemoryBank


# Rule types the UI can offer when adding a new reward rule.
RULE_TYPES = [
    "constant",
    "monotonic_progress",
    "signal_delta",
    "equals_terminal",
    "decrement_terminal",
    "stuck",
    "timeout",
]


class RewardSpec:
    """Editable container for a declarative rewards.json document."""

    def __init__(self, data: Dict[str, Any], source_path: Optional[str] = None):
        # Deep-copy so UI edits do not mutate shared pack caches.
        self.data = copy.deepcopy(data)
        # Path used when saving from the trainer UI.
        self.source_path = source_path

    @classmethod
    def from_file(cls, path: str) -> "RewardSpec":
        """Load a rewards document from disk."""
        # Open the declarative reward file.
        with open(path, "r", encoding="utf-8") as handle:
            # Parse JSON.
            data = json.load(handle)
        # Return a mutable spec wrapper.
        return cls(data, source_path=os.path.abspath(path))

    def save(self, path: Optional[str] = None) -> str:
        """Persist the rewards document; return the path written."""
        # Prefer explicit path, else rewrite the loaded file.
        target = path or self.source_path
        # Require a destination.
        if not target:
            # First-time saves must pick a path.
            raise ValueError("No path provided to save reward spec")
        # Create parent folders when needed.
        os.makedirs(os.path.dirname(os.path.abspath(target)) or ".", exist_ok=True)
        # Write pretty JSON for hand editing.
        with open(target, "w", encoding="utf-8") as handle:
            # Indent for readability in the UI / git diffs.
            json.dump(self.data, handle, indent=2)
            # Trailing newline.
            handle.write("\n")
        # Remember where we saved.
        self.source_path = os.path.abspath(target)
        # Return the absolute path.
        return self.source_path

    @property
    def rules(self) -> List[Dict[str, Any]]:
        """Return the mutable list of reward rules."""
        # Ensure the rules key always exists.
        if "rules" not in self.data:
            # Start with an empty rule list.
            self.data["rules"] = []
        # Return the live list for UI edits.
        return self.data["rules"]

    def enabled_rules(self) -> List[Dict[str, Any]]:
        """Return only rules that are enabled (default True)."""
        # Filter out explicitly disabled rules.
        return [rule for rule in self.rules if rule.get("enabled", True)]


class DeclarativeRewardEngine:
    """Interpret declarative reward rules against a MemoryBank + PyBoy memory."""

    def __init__(self, reward_spec: RewardSpec, memory_bank: MemoryBank):
        # Store the declarative reward document.
        self.spec = reward_spec
        # Store the named-address bank used to resolve signals.
        self.memory_bank = memory_bank
        # Episode trackers keyed by rule id / signal name.
        self._prev_values: Dict[str, float] = {}
        # Furthest progress values for monotonic rules.
        self._best_values: Dict[str, float] = {}
        # Direction history for oscillation detection per progress rule.
        self._directions: Dict[str, List[int]] = {}
        # Stuck counters per stuck rule.
        self._stuck_steps: Dict[str, int] = {}
        # Previous motion markers for stuck rules.
        self._motion_markers: Dict[str, Optional[float]] = {}

    def reset(self, memory) -> None:
        """Seed trackers from the current memory snapshot after env reset."""
        # Clear previous-value baselines.
        self._prev_values = {}
        # Clear furthest-progress baselines.
        self._best_values = {}
        # Clear oscillation windows.
        self._directions = {}
        # Clear stuck counters.
        self._stuck_steps = {}
        # Clear motion markers.
        self._motion_markers = {}
        # Seed baselines for every enabled rule that needs history.
        for rule in self.spec.enabled_rules():
            # Rule type selects which signals to seed.
            rule_type = rule.get("type")
            # Rule id namespaces tracker keys.
            rule_id = rule.get("id", rule_type)
            # Progress / decrement / delta rules need a starting value.
            if rule_type in (
                "monotonic_progress",
                "decrement_terminal",
                "signal_delta",
            ):
                # Resolve the watched signal name.
                signal = rule["signal"]
                # Read the current decoded value.
                value = float(self.memory_bank.read(memory, signal))
                # Seed previous sample.
                self._prev_values[rule_id] = value
                # Seed furthest progress for monotonic rules.
                if rule_type == "monotonic_progress":
                    # Episode-best starts at spawn progress.
                    self._best_values[rule_id] = value
                    # Empty oscillation window.
                    self._directions[rule_id] = []
            # Stuck rules seed the motion marker.
            if rule_type == "stuck":
                # Read the motion signal.
                marker = float(self.memory_bank.read(memory, rule["signal"]))
                # Store as baseline so the first step is not stuck.
                self._motion_markers[rule_id] = marker
                # Reset consecutive stuck count.
                self._stuck_steps[rule_id] = 0

    def step(self, memory, episode_length: int) -> Tuple[float, bool, bool, Dict[str, Any]]:
        """Evaluate all enabled rules for one env step."""
        # Accumulate shaped reward.
        reward = 0.0
        # Per-rule component breakdown for UI / logs.
        components: Dict[str, float] = {}
        # True task failure / success ends.
        terminated = False
        # Budget / stuck ends.
        truncated = False
        # Extra diagnostic fields exposed on the info dict.
        extras: Dict[str, Any] = {}

        # Evaluate each enabled rule in declaration order.
        for rule in self.spec.enabled_rules():
            # Dispatch by declarative type.
            rule_type = rule.get("type")
            # Stable id for components / trackers.
            rule_id = rule.get("id", rule_type)
            # Constant per-step cost or bonus.
            if rule_type == "constant":
                # Read the constant value.
                value = float(rule.get("value", 0.0))
                # Add to the step reward.
                reward += value
                # Record the component.
                components[rule_id] = value
            # Monotonic progress with anti-wiggle shaping.
            elif rule_type == "monotonic_progress":
                # Apply progress shaping and merge components.
                delta_reward, progress_bits, progress_extras = self._monotonic_progress(
                    rule, rule_id, memory
                )
                # Add shaped progress reward.
                reward += delta_reward
                # Merge component keys.
                components.update(progress_bits)
                # Merge diagnostics (progress pct, best, etc.).
                extras.update(progress_extras)
            # Dense reward from per-step signal deltas (e.g. fighting-game HP).
            elif rule_type == "signal_delta":
                # Read the watched signal.
                current = float(self.memory_bank.read(memory, rule["signal"]))
                # Previous baseline (seeded on reset, else current).
                previous = float(self._prev_values.get(rule_id, current))
                # Signed change this step.
                delta = current - previous
                # Accumulated reward for this rule.
                delta_reward = 0.0
                # Reward / penalise increases.
                if delta > 0.0:
                    # Per-unit weight for upward moves.
                    delta_reward += delta * float(rule.get("increase_scale", 0.0))
                # Reward / penalise decreases.
                elif delta < 0.0:
                    # Per-unit weight for downward moves (damage dealt uses >0).
                    delta_reward += (-delta) * float(rule.get("decrease_scale", 0.0))
                # Apply when non-zero.
                if delta_reward != 0.0:
                    # Add to the step reward.
                    reward += delta_reward
                    # Record component.
                    components[rule_id] = delta_reward
                # Advance baseline after applying the delta.
                self._prev_values[rule_id] = current
            # Terminal when a signal equals a sentinel.
            elif rule_type == "equals_terminal":
                # Optional grace so menu screens with 0 HP do not end episodes.
                min_len = int(rule.get("min_episode_length", 0))
                # Skip until the episode is long enough.
                if episode_length < min_len:
                    # Do not evaluate this terminal yet.
                    pass
                else:
                    # Current decoded signal.
                    current = float(self.memory_bank.read(memory, rule["signal"]))
                    # Target sentinel value.
                    target = float(rule.get("equals", 0))
                    # Fire when equal.
                    if current == target:
                        # Terminal bonus / penalty.
                        value = float(rule.get("reward", 0.0))
                        # Optionally replace the whole step reward.
                        if rule.get("replace_reward", False):
                            # Death-style replace.
                            reward = value
                        else:
                            # Additive terminal bonus.
                            reward += value
                        # Record component.
                        components[rule_id] = value
                        # Mark episode terminated when requested.
                        if rule.get("terminal", True):
                            # Success / event terminal.
                            terminated = True
            # Terminal when a counter decreases (lives, etc.).
            elif rule_type == "decrement_terminal":
                # Optional grace period before decrement terminals apply.
                min_len = int(rule.get("min_episode_length", 0))
                # Current counter value.
                current = float(self.memory_bank.read(memory, rule["signal"]))
                # Previous baseline from reset / last step.
                previous = self._prev_values.get(rule_id, current)
                # Detect a drop after the grace window.
                if episode_length >= min_len and current < previous:
                    # Terminal penalty / bonus.
                    value = float(rule.get("reward", 0.0))
                    # Replace or add.
                    if rule.get("replace_reward", True):
                        # Death usually replaces other shaping.
                        reward = value
                    else:
                        # Additive variant.
                        reward += value
                    # Record component.
                    components[rule_id] = value
                    # Terminate when configured.
                    if rule.get("terminal", True):
                        # End the episode.
                        terminated = True
                # Always advance the baseline after the check.
                self._prev_values[rule_id] = current
            # Stuck detection via unchanged motion marker.
            elif rule_type == "stuck":
                # Evaluate stuck logic.
                stuck_reward, stuck_trunc, stuck_bits = self._stuck_rule(
                    rule, rule_id, memory, episode_length
                )
                # Add stuck penalty.
                reward += stuck_reward
                # Merge components.
                components.update(stuck_bits)
                # Truncate when the limit is hit.
                if stuck_trunc:
                    # Mark truncated.
                    truncated = True
            # Progressive timeout (+ optional critical timer signal).
            elif rule_type == "timeout":
                # Decide whether the budget / timer fired.
                timed_out = self._timeout_fired(rule, memory, episode_length)
                # Apply only when not already terminated.
                if timed_out and not terminated:
                    # Timeout penalty.
                    value = float(rule.get("penalty", 0.0))
                    # Add penalty.
                    reward += value
                    # Record component.
                    components[rule_id] = value
                    # Truncate when configured.
                    if rule.get("truncate", True):
                        # Soft episode end.
                        truncated = True
            else:
                # Unknown rule types should not silently no-op in training.
                raise ValueError(f"Unsupported reward rule type '{rule_type}'")

        # Clip bounds from the top-level reward spec.
        clip_min = float(self.spec.data.get("clip_min", -50.0))
        # Upper clip bound.
        clip_max = float(self.spec.data.get("clip_max", 50.0))
        # Clip the final scalar reward.
        reward = float(np.clip(reward, clip_min, clip_max))
        # Build the info payload for callbacks / UI.
        info = {
            "reward_components": components,
            **extras,
        }
        # Return Gymnasium-style shaped outputs.
        return reward, terminated, truncated, info

    def _monotonic_progress(self, rule, rule_id, memory):
        """Reward only new furthest progress; punish retreat and oscillation."""
        # Read the progress signal (usually 0-100).
        progress = float(self.memory_bank.read(memory, rule["signal"]))
        # Previous progress sample.
        prev = float(self._prev_values.get(rule_id, progress))
        # Episode furthest progress.
        best = float(self._best_values.get(rule_id, progress))
        # Step delta (can re-cross old ground).
        delta = progress - prev
        # Newly claimed territory beyond the episode best.
        new_territory = progress - best
        # Accumulated reward for this rule family.
        reward = 0.0
        # Component breakdown.
        components: Dict[str, float] = {}
        # Pay only for new furthest progress.
        if new_territory > 0.0:
            # Scale newly claimed progress.
            progress_reward = new_territory * float(rule.get("scale", 1.0))
            # Add dense progress.
            reward += progress_reward
            # Record progress component.
            components[f"{rule_id}_progress"] = progress_reward
            # Flat bonus for a new best.
            bonus = float(rule.get("new_best_bonus", 0.0))
            # Add new-best bonus.
            reward += bonus
            # Record bonus when non-zero.
            if bonus:
                # Store under a stable key.
                components[f"{rule_id}_new_best"] = bonus
            # Update furthest progress.
            self._best_values[rule_id] = progress
            # Clear oscillation history after genuine progress.
            self._directions[rule_id] = []
        elif delta < 0.0:
            # Backward penalty.
            backward = float(rule.get("backward_penalty", 0.0))
            # Apply retreat penalty.
            reward += backward
            # Record backward component.
            components[f"{rule_id}_backward"] = backward
            # Also apply no-forward while retreating.
            no_forward = float(rule.get("no_forward_penalty", 0.0))
            # Apply stall penalty.
            reward += no_forward
            # Record stall component.
            components[f"{rule_id}_no_forward"] = no_forward
        else:
            # Standing still or re-entering visited ground.
            no_forward = float(rule.get("no_forward_penalty", 0.0))
            # Apply stall / re-cross penalty.
            reward += no_forward
            # Record stall component.
            components[f"{rule_id}_no_forward"] = no_forward

        # Direction sign for oscillation tracking.
        if delta > 0.0:
            # Forward step.
            direction = 1
        elif delta < 0.0:
            # Backward step.
            direction = -1
        else:
            # Flat reading.
            direction = 0
        # Append non-zero directions into the window.
        if direction != 0:
            # Fetch / create the window list.
            window_list = self._directions.setdefault(rule_id, [])
            # Append this sign.
            window_list.append(direction)
            # Configured window length.
            window = int(rule.get("oscillation_window", 6))
            # Trim to the window.
            self._directions[rule_id] = window_list[-window:]
        # Oscillation penalty when wiggling without new territory.
        osc_window = int(rule.get("oscillation_window", 6))
        # Recent signs.
        recent = self._directions.get(rule_id, [])
        # Fire when the window is full and flipping often.
        if (
            new_territory <= 0.0
            and len(recent) >= osc_window
            and self._is_oscillating(recent)
        ):
            # Oscillation penalty.
            osc = float(rule.get("oscillation_penalty", 0.0))
            # Apply penalty.
            reward += osc
            # Record component.
            components[f"{rule_id}_oscillation"] = osc
        # Advance previous progress.
        self._prev_values[rule_id] = progress
        # Diagnostics for UI / callbacks.
        extras = {
            "progress_pct": progress,
            "best_progress_pct": float(self._best_values.get(rule_id, progress)),
        }
        # Return shaped pieces.
        return reward, components, extras

    def _stuck_rule(self, rule, rule_id, memory, episode_length):
        """Penalise unchanged motion markers and optionally truncate."""
        # Current motion marker.
        marker = float(self.memory_bank.read(memory, rule["signal"]))
        # Previous marker from reset / last step.
        previous = self._motion_markers.get(rule_id)
        # First observation after reset is never stuck.
        if previous is None:
            # Seed marker.
            self._motion_markers[rule_id] = marker
            # No penalty yet.
            return 0.0, False, {}
        # Unchanged marker means stuck.
        is_stuck = marker == previous
        # Update cached marker.
        self._motion_markers[rule_id] = marker
        # Grace period before stuck penalties apply.
        grace = int(rule.get("grace_steps", 100))
        # Truncation limit for consecutive stuck steps.
        limit = int(rule.get("limit_steps", 800))
        # Reward / truncate outputs.
        reward = 0.0
        # Truncation flag.
        truncated = False
        # Component dict.
        components: Dict[str, float] = {}
        # Apply only after the grace window.
        if episode_length > grace and is_stuck:
            # Increment consecutive stuck count.
            self._stuck_steps[rule_id] = self._stuck_steps.get(rule_id, 0) + 1
            # Per-step stuck penalty.
            penalty = float(rule.get("penalty", 0.0))
            # Apply penalty.
            reward += penalty
            # Record component.
            components[rule_id] = penalty
            # Truncate when over the limit.
            if self._stuck_steps[rule_id] >= limit and rule.get("truncate", True):
                # Soft end.
                truncated = True
        else:
            # Moving again clears the counter.
            self._stuck_steps[rule_id] = 0
        # Expose stuck steps on extras via components only; env can read tracker.
        return reward, truncated, components

    def _timeout_fired(self, rule, memory, episode_length) -> bool:
        """Return True when step budget or optional critical signal fires."""
        # Optional progress signal scales the timeout threshold.
        progress_signal = rule.get("progress_signal")
        # Progress factor in [0, 1].
        progress_factor = 0.0
        # Read progress when configured.
        if progress_signal:
            # Decode progress percent.
            progress_pct = float(self.memory_bank.read(memory, progress_signal))
            # Normalize assuming 0-100 progress signals.
            progress_factor = max(0.0, min(1.0, progress_pct / 100.0))
        # Base step budget.
        base_steps = int(rule.get("base_steps", 2500))
        # Max step budget near stage end.
        max_steps = int(rule.get("max_steps", base_steps))
        # Interpolate threshold.
        threshold = int(base_steps + (max_steps - base_steps) * progress_factor)
        # Step-budget timeout.
        if episode_length >= threshold:
            # Budget exhausted.
            return True
        # Optional boolean-like critical timer signal.
        also_signal = rule.get("also_signal")
        # When present, treat >0.5 as critical.
        if also_signal:
            # Read the critical signal (0/1).
            critical = float(self.memory_bank.read(memory, also_signal))
            # Fire when critical.
            if critical >= 0.5:
                # In-game timer exhausted.
                return True
        # No timeout this step.
        return False

    @staticmethod
    def _is_oscillating(directions) -> bool:
        """Return True when recent signed moves flip direction repeatedly."""
        # Need at least two signs.
        if len(directions) < 2:
            # Not enough history.
            return False
        # Count sign changes.
        flips = 0
        # Walk adjacent pairs.
        for index in range(1, len(directions)):
            # Flip when consecutive signs disagree.
            if directions[index] != directions[index - 1]:
                # Count the flip.
                flips += 1
        # Many flips inside the window => oscillation farming.
        return flips >= (len(directions) // 2)
