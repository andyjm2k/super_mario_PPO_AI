"""Field schemas for the declarative reward rule editor."""

from pdt.reward_engine import RULE_TYPES

# Re-export for UI imports.
__all__ = ["RULE_TYPES", "RULE_FIELDS", "default_rule_for_type"]

# Fields shown in the editor for each rule type (label, key, kind).
RULE_FIELDS = {
    "constant": [("VALUE", "value", "float")],
    "monotonic_progress": [
        ("SIGNAL", "signal", "signal"),
        ("SCALE", "scale", "float"),
        ("NEW BEST BONUS", "new_best_bonus", "float"),
        ("BACKWARD PENALTY", "backward_penalty", "float"),
        ("NO FORWARD", "no_forward_penalty", "float"),
        ("OSC PENALTY", "oscillation_penalty", "float"),
        ("OSC WINDOW", "oscillation_window", "int"),
    ],
    "signal_delta": [
        ("SIGNAL", "signal", "signal"),
        ("DECREASE SCALE", "decrease_scale", "float"),
        ("INCREASE SCALE", "increase_scale", "float"),
    ],
    "equals_terminal": [
        ("SIGNAL", "signal", "signal"),
        ("EQUALS", "equals", "float"),
        ("REWARD", "reward", "float"),
        ("TERMINAL", "terminal", "bool"),
        ("REPLACE REWARD", "replace_reward", "bool"),
        ("MIN EP LENGTH", "min_episode_length", "int"),
    ],
    "decrement_terminal": [
        ("SIGNAL", "signal", "signal"),
        ("REWARD", "reward", "float"),
        ("TERMINAL", "terminal", "bool"),
        ("REPLACE REWARD", "replace_reward", "bool"),
        ("MIN EP LENGTH", "min_episode_length", "int"),
    ],
    "stuck": [
        ("SIGNAL", "signal", "signal"),
        ("PENALTY", "penalty", "float"),
        ("GRACE STEPS", "grace_steps", "int"),
        ("LIMIT STEPS", "limit_steps", "int"),
        ("TRUNCATE", "truncate", "bool"),
    ],
    "timeout": [
        ("BASE STEPS", "base_steps", "int"),
        ("MAX STEPS", "max_steps", "int"),
        ("PROGRESS SIGNAL", "progress_signal", "signal"),
        ("ALSO SIGNAL", "also_signal", "signal"),
        ("PENALTY", "penalty", "float"),
        ("TRUNCATE", "truncate", "bool"),
    ],
}


def default_rule_for_type(rule_type, signal_names, index):
    """Return a new rule dict skeleton for the given declarative type."""
    # Base fields shared by every rule.
    new_rule = {
        "id": f"{rule_type}_{index}",
        "type": rule_type,
        "enabled": True,
    }
    # First known signal name for signal-backed rules.
    first_signal = signal_names[0] if signal_names else ""
    # Type-specific defaults.
    if rule_type == "constant":
        # Default living-style cost.
        new_rule["value"] = -0.01
    elif rule_type == "monotonic_progress":
        # Progress defaults.
        new_rule.update(
            {
                "signal": first_signal,
                "scale": 1.0,
                "new_best_bonus": 0.0,
                "backward_penalty": 0.0,
                "no_forward_penalty": 0.0,
                "oscillation_penalty": 0.0,
                "oscillation_window": 6,
            }
        )
    elif rule_type == "signal_delta":
        # Delta shaping defaults (e.g. HP damage).
        new_rule.update(
            {
                "signal": first_signal,
                "decrease_scale": 1.0,
                "increase_scale": 0.0,
            }
        )
    elif rule_type == "equals_terminal":
        # Equals terminal defaults.
        new_rule.update(
            {
                "signal": first_signal,
                "equals": 0,
                "reward": 0.0,
                "terminal": True,
                "replace_reward": False,
            }
        )
    elif rule_type == "decrement_terminal":
        # Decrement terminal defaults.
        new_rule.update(
            {
                "signal": first_signal,
                "reward": -10.0,
                "terminal": True,
                "replace_reward": True,
            }
        )
    elif rule_type == "stuck":
        # Stuck defaults.
        new_rule.update(
            {
                "signal": first_signal,
                "penalty": -0.1,
                "grace_steps": 100,
                "limit_steps": 800,
                "truncate": True,
            }
        )
    elif rule_type == "timeout":
        # Timeout defaults.
        new_rule.update(
            {
                "base_steps": 2500,
                "max_steps": 12000,
                "penalty": -10.0,
                "truncate": True,
            }
        )
    # Return the skeleton.
    return new_rule
