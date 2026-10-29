"""Shared emulator backend protocol for PyBoy and PyBoyAdvance."""

from __future__ import annotations

from typing import Any, List, Protocol, runtime_checkable


# Sentinel path meaning "boot the ROM without loading a snapshot".
BOOT_STATE = "__boot__"


@runtime_checkable
class EmulatorBackend(Protocol):
    """Minimal surface the Gym env needs from any emulator engine."""

    # Memory-like object supporting ``memory[address] -> int``.
    memory: Any

    def tick(self, count: int = 1) -> None:
        """Advance the emulator by ``count`` rendered frames."""

    def press_button(self, name: str) -> None:
        """Hold a button identified by a pack action name (a, left, l, …)."""

    def release_button(self, name: str) -> None:
        """Release a button identified by a pack action name."""

    def release_all(self) -> None:
        """Release every commonly used button."""

    def get_rgb_frame(self):
        """Return an RGB uint8 ndarray of the current screen."""

    def set_emulation_speed(self, speed: float) -> None:
        """Set speed multiplier (0 = unlimited)."""

    def load_state_file(self, path: str) -> None:
        """Restore a start snapshot, or no-op for BOOT_STATE."""

    def stop(self) -> None:
        """Release emulator resources when possible."""
