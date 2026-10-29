"""PyBoy (GB/GBC) emulator backend."""

from __future__ import annotations

import os

import numpy as np
from pyboy import PyBoy

from pdt.emulator import BOOT_STATE


class PyBoyBackend:
    """Thin adapter around PyBoy matching the shared emulator protocol."""

    def __init__(
        self,
        rom_path: str,
        window: str = "null",
        emulation_speed: float = 0,
    ):
        # Absolute ROM path for PyBoy.
        self.rom_path = rom_path
        # Window mode remembered for recycle recreates.
        self.window = window
        # Boot the GB/GBC emulator.
        self._emu = PyBoy(rom_path, window=window)
        # Apply initial speed target.
        self.set_emulation_speed(emulation_speed)

    @property
    def memory(self):
        """Expose PyBoy's addressable memory mapping."""
        # Direct passthrough; supports memory[addr].
        return self._emu.memory

    def tick(self, count: int = 1) -> None:
        """Advance ``count`` emulator frames."""
        # Tick one frame at a time (PyBoy API).
        for _ in range(max(1, int(count))):
            # Advance a single frame.
            self._emu.tick()

    def press_button(self, name: str) -> None:
        """Press a PyBoy button by lowercase name."""
        # Normalize pack button ids.
        self._emu.button_press(str(name).lower())

    def release_button(self, name: str) -> None:
        """Release a PyBoy button by lowercase name."""
        # Normalize pack button ids.
        self._emu.button_release(str(name).lower())

    def release_all(self) -> None:
        """Release the standard GB/GBC button set."""
        # D-pad + face + menu buttons.
        for name in ("left", "right", "up", "down", "a", "b", "start", "select"):
            # Best-effort release.
            self._emu.button_release(name)

    def get_rgb_frame(self):
        """Return the current screen as RGB uint8."""
        # Convert PIL image and drop alpha if present.
        return np.array(self._emu.screen.image)[:, :, :3].astype(np.uint8)

    def set_emulation_speed(self, speed: float) -> None:
        """Set PyBoy speed (0 = unlimited)."""
        # Best-effort for API differences.
        try:
            # PyBoy expects an int multiplier.
            self._emu.set_emulation_speed(int(speed))
        except Exception:
            # Ignore missing helpers.
            pass

    def load_state_file(self, path: str) -> None:
        """Load a PyBoy .state file, or skip for boot-only starts."""
        # Boot sentinel means no snapshot restore.
        if path == BOOT_STATE:
            # Nothing to load.
            return
        # Require a real file on disk.
        if not os.path.isfile(path):
            # Fail fast with a clear path.
            raise FileNotFoundError(f"PyBoy state not found: {path}")
        # Open the binary snapshot.
        with open(path, "rb") as handle:
            try:
                # Restore PyBoy state.
                self._emu.load_state(handle)
            except Exception as exc:
                # Version mismatches are common across PyBoy minors.
                raise RuntimeError(
                    f"Failed to load PyBoy state '{path}'. "
                    "State files must match the installed PyBoy version "
                    "(this project pins pyboy==2.6.0)."
                ) from exc

    def stop(self) -> None:
        """Stop the PyBoy instance."""
        # Clean shutdown.
        self._emu.stop()
