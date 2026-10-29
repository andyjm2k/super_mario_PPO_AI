"""Load and decode named memory signals from a game pack memory.json."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional


def parse_address(value: Any) -> int:
    """Parse an int or hex string (e.g. '0xC24B') into an integer address."""
    # Already-numeric addresses pass through unchanged.
    if isinstance(value, int):
        # Return the integer address.
        return value
    # Convert string addresses, supporting 0x-prefixed hex.
    text = str(value).strip()
    # Hex form used in exported memory maps.
    if text.lower().startswith("0x"):
        # Parse base-16.
        return int(text, 16)
    # Decimal fallback for plain integers as strings.
    return int(text)


def tile_to_digit(tile_value: int, tile_base: int, max_digit: int) -> int:
    """Map a HUD tile id onto a decimal digit, else 0."""
    # Reject tiles outside the configured digit range.
    if tile_value < tile_base or tile_value > tile_base + max_digit:
        # Unknown tiles become zero.
        return 0
    # Offset from the digit-0 tile.
    return tile_value - tile_base


def digits_to_int(digits) -> int:
    """Assemble digit tiles into an integer, skipping leading zeros."""
    # Default when every digit is blank/zero.
    score = 0
    # Walk until the first non-zero digit.
    for index, value in enumerate(digits):
        # Skip leading zeros so "00123" becomes 123.
        if value > 0:
            # Concatenate the remaining digits.
            score = int("".join(map(str, digits[index:])))
            # Stop after the first meaningful group.
            break
    # Return the assembled integer.
    return score


class MemoryBank:
    """Named signal reader backed by a declarative memory.json file."""

    def __init__(self, spec: Dict[str, Any], source_path: Optional[str] = None):
        # Store the full declarative spec.
        self.spec = spec
        # Remember where the file was loaded from (for UI save/import).
        self.source_path = source_path
        # Per-signal definitions keyed by name.
        self.signals: Dict[str, Any] = dict(spec.get("signals", {}))
        # Pack-level decoder defaults.
        self.defaults: Dict[str, Any] = dict(spec.get("defaults", {}))

    @classmethod
    def from_file(cls, path: str) -> "MemoryBank":
        """Load a memory bank JSON from disk."""
        # Open the declarative memory map.
        with open(path, "r", encoding="utf-8") as handle:
            # Parse JSON into a dict.
            data = json.load(handle)
        # Construct the bank with the source path attached.
        return cls(data, source_path=os.path.abspath(path))

    def save(self, path: Optional[str] = None) -> str:
        """Write the current memory spec back to disk; return the path used."""
        # Prefer an explicit path, else rewrite the original file.
        target = path or self.source_path
        # Refuse to save when no destination is known.
        if not target:
            # Caller must provide a path on first save.
            raise ValueError("No path provided to save memory bank")
        # Ensure parent directories exist.
        os.makedirs(os.path.dirname(os.path.abspath(target)) or ".", exist_ok=True)
        # Serialize with stable formatting for diffs / hand edits.
        with open(target, "w", encoding="utf-8") as handle:
            # Pretty-print for human editing in the UI / text editors.
            json.dump(self.spec, handle, indent=2)
            # Trailing newline keeps editors happy.
            handle.write("\n")
        # Remember the path we wrote.
        self.source_path = os.path.abspath(target)
        # Return the absolute path for UI status text.
        return self.source_path

    def signal_names(self):
        """Return sorted signal names for UI dropdowns."""
        # Alphabetical list of declarative signal keys.
        return sorted(self.signals.keys())

    def _signal_param(self, signal: Dict[str, Any], key: str, default=None):
        """Resolve a signal field, falling back to pack defaults."""
        # Prefer the per-signal override.
        if key in signal:
            # Use the signal-local value.
            return signal[key]
        # Fall back to pack-level defaults when present.
        if key in self.defaults:
            # Use the shared default.
            return self.defaults[key]
        # Final fallback supplied by the caller.
        return default

    def read(self, memory, name: str) -> float:
        """Decode one named signal from a PyBoy-like memory mapping."""
        # Unknown signal names fail fast so packs are easy to debug.
        if name not in self.signals:
            # Tell the user which key is missing.
            raise KeyError(f"Unknown memory signal '{name}'")
        # Fetch the declarative signal definition.
        signal = self.signals[name]
        # Decoder name selects how bytes become a float/int.
        decode = str(signal.get("decode", "raw")).lower()
        # Raw single-byte read.
        if decode == "raw":
            # Parse the single address.
            address = parse_address(signal["address"])
            # Return the unsigned byte as a float for the reward engine.
            return float(memory[address])
        # Progress tile mapped onto 0-100 percent.
        if decode == "tile_percent":
            # Progress tile address.
            address = parse_address(signal["address"])
            # Digit-0 tile id for this HUD font.
            tile_base = int(self._signal_param(signal, "tile_digit_base", 208))
            # Maximum progress segment observed for this game.
            max_tile = int(self._signal_param(signal, "max_progress_tile", 12))
            # Convert tile -> segment index.
            segment = tile_to_digit(int(memory[address]), tile_base, max_tile)
            # Normalize to a percentage.
            return (segment / float(max_tile)) * 100.0
        # Multi-digit HUD score assembly.
        if decode == "digit_tiles_int":
            # List of digit addresses in MSD->LSD order.
            addresses = [parse_address(item) for item in signal["addresses"]]
            # Digit-0 tile id.
            tile_base = int(self._signal_param(signal, "tile_digit_base", 208))
            # Maximum digit tile offset allowed.
            max_digit = int(self._signal_param(signal, "max_digit", 12))
            # Decode each digit tile.
            digits = [
                tile_to_digit(int(memory[addr]), tile_base, max_digit)
                for addr in addresses
            ]
            # Assemble into an integer score.
            return float(digits_to_int(digits))
        # Timer-critical heuristic used by Mario Deluxe (shows 001).
        if decode == "timer_critical":
            # Three timer digit addresses.
            addresses = [parse_address(item) for item in signal["addresses"]]
            # Digit-0 tile id.
            tile_base = int(self._signal_param(signal, "tile_digit_base", 208))
            # Read the three tiles.
            hundreds = int(memory[addresses[0]])
            # Tens digit tile.
            tens = int(memory[addresses[1]])
            # Ones digit tile.
            ones = int(memory[addresses[2]])
            # Match the historical "001" critical pattern.
            critical = (
                hundreds == tile_base
                and tens == tile_base
                and ones == tile_base + 1
            )
            # Return 1.0 / 0.0 so rules can treat it like a boolean signal.
            return 1.0 if critical else 0.0
        # Reject unknown decoder names so pack authors get a clear error.
        raise ValueError(f"Unsupported decode mode '{decode}' for signal '{name}'")

    def merge_import(self, other: "MemoryBank", overwrite: bool = True) -> int:
        """Merge signals from another bank; return count of signals added/updated."""
        # Count how many signals change.
        changed = 0
        # Walk every imported signal name.
        for name, definition in other.signals.items():
            # Skip existing keys when overwrite is disabled.
            if name in self.signals and not overwrite:
                # Leave the local definition alone.
                continue
            # Copy the imported definition.
            self.signals[name] = dict(definition)
            # Track the mutation.
            changed += 1
        # Keep the top-level spec in sync for saving.
        self.spec["signals"] = self.signals
        # Merge defaults when the import provides them.
        for key, value in other.defaults.items():
            # Only fill missing defaults unless overwrite is on.
            if overwrite or key not in self.defaults:
                # Store the default.
                self.defaults[key] = value
        # Persist defaults onto the spec dict.
        self.spec["defaults"] = self.defaults
        # Return how many signal entries changed.
        return changed
