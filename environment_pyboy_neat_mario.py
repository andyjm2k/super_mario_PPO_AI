"""Backwards-compatible Mario env import — use pdt.pyboy_env.PyBoyEnv."""

from pdt.pyboy_env import GbaGame, PyBoyEnv

# Re-export historical names expected by older tests / scripts.
__all__ = ["GbaGame", "PyBoyEnv", "MarioDeluxeEnv"]

# Alias used in older documentation.
MarioDeluxeEnv = PyBoyEnv
