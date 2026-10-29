"""Factory that builds the correct emulator backend for a game pack."""

from __future__ import annotations

from typing import Optional

from pdt.emulator.pyboy_backend import PyBoyBackend


def normalize_engine_name(name: str) -> str:
    """Normalize game.json emulator ids onto canonical backend keys."""
    # Lowercase for comparisons.
    key = str(name or "pyboy").strip().lower().replace("-", "_")
    # Aliases for PyBoy (GB/GBC).
    if key in ("pyboy", "gb", "gbc", "gameboy"):
        # Canonical PyBoy id.
        return "pyboy"
    # Aliases for PyBoyAdvance (GBA RAM snapshots, frozen preview frames).
    if key in ("pyboy_advance", "pyboyadvance", "gba", "advance"):
        # Canonical Advance id.
        return "pyboy_advance"
    # Aliases for headless mGBA (live frames + full savestates).
    if key in ("mgba", "libmgba", "lib_mgba"):
        # Canonical mGBA id.
        return "mgba"
    # Unknown engines fail fast.
    raise ValueError(
        f"Unsupported emulator engine '{name}'. "
        "Use 'pyboy' (GB/GBC), 'pyboy_advance' (GBA), or 'mgba' (GBA live frames)."
    )


def create_emulator_backend(
    engine: str,
    rom_path: str,
    window: str = "null",
    emulation_speed: float = 0,
    bios_path: Optional[str] = None,
    skip_bios: bool = True,
):
    """Construct a backend instance for the requested engine."""
    # Canonicalize the engine name.
    key = normalize_engine_name(engine)
    # GB/GBC path.
    if key == "pyboy":
        # Build the PyBoy adapter.
        return PyBoyBackend(
            rom_path=rom_path,
            window=window,
            emulation_speed=emulation_speed,
        )
    # Headless mGBA path (lazy-imports mgba inside the backend).
    if key == "mgba":
        # Import here so a missing build does not break PyBoy packs.
        from pdt.emulator.mgba_backend import MgbaBackend

        # Build the mGBA adapter (live framebuffer + full savestates).
        return MgbaBackend(
            rom_path=rom_path,
            bios_path=bios_path,
            skip_bios=skip_bios,
            emulation_speed=emulation_speed,
            window=window,
        )
    # GBA path (lazy-imports pyboy-advance inside the backend).
    from pdt.emulator.pyboy_advance_backend import PyBoyAdvanceBackend

    # Build the Advance adapter.
    return PyBoyAdvanceBackend(
        rom_path=rom_path,
        bios_path=bios_path,
        skip_bios=skip_bios,
        emulation_speed=emulation_speed,
        window=window,
    )
