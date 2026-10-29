"""Discover and load per-game packs under the games/ directory."""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from typing import List, Optional

from pdt.emulator import BOOT_STATE
from pdt.memory_bank import MemoryBank
from pdt.reward_engine import RewardSpec


# Default root that holds one folder per game.
DEFAULT_GAMES_ROOT = "games"


@dataclass
class GamePack:
    """Resolved paths and metadata for one trainable PyBoy game."""

    # Stable id matching the folder / game.json id.
    game_id: str
    # Human-readable display name for the UI.
    name: str
    # Absolute path to the game pack folder.
    root_dir: str
    # Absolute path to the ROM file.
    rom_path: str
    # Absolute path to the states directory.
    states_dir: str
    # Absolute path to memory.json.
    memory_path: str
    # Absolute path to rewards.json.
    rewards_path: str
    # Parsed game.json body.
    meta: dict

    def load_memory(self) -> MemoryBank:
        """Load the pack's memory address bank."""
        # Delegate to the MemoryBank file loader.
        return MemoryBank.from_file(self.memory_path)

    def load_rewards(self) -> RewardSpec:
        """Load the pack's declarative reward document."""
        # Delegate to the RewardSpec file loader.
        return RewardSpec.from_file(self.rewards_path)

    def discover_states(self) -> List[str]:
        """Return start snapshots for this pack (engine-specific extensions)."""
        # Prefer ROM boot when the pack opts in (needed for live GBA screens).
        if self.prefer_boot_start:
            # Ignore on-disk snapshots until PyBoyAdvance has real savestates.
            return [BOOT_STATE]
        # Extensions declared in game.json, with sensible defaults.
        extensions = list(self.meta.get("state_extensions", self._default_state_extensions()))
        # Accumulate matching files.
        paths = []
        # Walk each configured extension.
        for ext in extensions:
            # Normalize leading dots.
            suffix = ext if str(ext).startswith(".") else f".{ext}"
            # Match files in the states directory.
            pattern = os.path.join(self.states_dir, f"*{suffix}")
            # Extend the curriculum list.
            paths.extend(glob.glob(pattern))
        # Sort for stable curricula / UI lists.
        paths = sorted({os.path.normpath(path) for path in paths})
        # Allow ROM-boot episodes when the pack opts in and no snapshots exist.
        if not paths and self.allow_boot_start:
            # Sentinel consumed by emulator backends.
            return [BOOT_STATE]
        # Return discovered paths (may be empty; env will error if required).
        return paths

    def _default_state_extensions(self) -> List[str]:
        """Snapshot suffixes used when game.json omits state_extensions."""
        # Compare a normalized engine id without importing the factory.
        engine = str(self.emulator_engine).strip().lower().replace("-", "_")
        # GB/GBC packs use PyBoy .state files.
        if engine in ("pyboy", "gb", "gbc", "gameboy"):
            # PyBoy savestates.
            return [".state"]
        # mGBA packs use full savestate blobs, not .pba_state pickles.
        if engine in ("mgba", "libmgba", "lib_mgba"):
            # Written by tools/save_mgba_state.py.
            return [".mgba_state"]
        # PyBoyAdvance and any other GBA id.
        return [".state", ".pba_state"]

    @property
    def emulator_engine(self) -> str:
        """Emulator backend id: pyboy, pyboy_advance, or mgba."""
        # Default remains PyBoy for existing GB/GBC packs.
        return str(self.meta.get("emulator", "pyboy"))

    @property
    def bios_path(self) -> Optional[str]:
        """Absolute BIOS path for GBA packs, else None."""
        # Relative bios path from game.json.
        bios_rel = self.meta.get("bios")
        # No BIOS configured.
        if not bios_rel:
            # GB/GBC packs do not need one.
            return None
        # Resolve against the pack root.
        return os.path.normpath(os.path.join(self.root_dir, bios_rel))

    @property
    def skip_bios(self) -> bool:
        """Whether PyBoyAdvance should skip the BIOS boot animation."""
        # Default True for faster episode starts.
        return bool(self.meta.get("skip_bios", True))

    @property
    def allow_boot_start(self) -> bool:
        """Allow training from ROM boot when no state files exist."""
        # Opt-in so accidental empty states folders still fail loudly.
        return bool(self.meta.get("allow_boot_start", False))

    @property
    def prefer_boot_start(self) -> bool:
        """Force ROM-boot starts even when snapshot files exist."""
        # Opt-in only: default GBA packs boot-then-apply ``*.pba_state``
        # (RAM + rgb_frame). Set true to ignore snapshots and stay on title.
        return bool(self.meta.get("prefer_boot_start", False))

    @property
    def obs_width(self) -> int:
        """CNN observation width from game.json."""
        # Read nested obs.width with a safe default.
        return int(self.meta.get("obs", {}).get("width", 80))

    @property
    def obs_height(self) -> int:
        """CNN observation height from game.json."""
        # Read nested obs.height with a safe default.
        return int(self.meta.get("obs", {}).get("height", 64))

    @property
    def crop_top(self) -> int:
        """Top-pixel HUD crop from game.json."""
        # Read nested obs.crop_top with a safe default.
        return int(self.meta.get("obs", {}).get("crop_top", 0))

    @property
    def actions(self) -> list:
        """Discrete action button maps from game.json."""
        # Each entry is {"id": ..., "buttons": [...]} .
        return list(self.meta.get("actions", [{"buttons": []}]))

    @property
    def wait_frames(self) -> int:
        """Frames to hold each action combo."""
        # Default to a single emulator tick.
        return int(self.meta.get("wait_frames", 1))


def list_game_ids(games_root: str = DEFAULT_GAMES_ROOT) -> List[str]:
    """Return sorted game folder names that contain a playable game.json + ROM."""
    # Nothing to list when the root is missing.
    if not os.path.isdir(games_root):
        # Empty catalogue.
        return []
    # Accumulate valid pack ids.
    ids = []
    # Walk immediate children of games/.
    for name in sorted(os.listdir(games_root)):
        # Candidate pack folder.
        folder = os.path.join(games_root, name)
        # Skip non-directories.
        if not os.path.isdir(folder):
            # Ignore loose files.
            continue
        # Manifest path.
        manifest = os.path.join(folder, "game.json")
        # Require a game.json manifest.
        if not os.path.isfile(manifest):
            # Incomplete scaffold.
            continue
        try:
            # Parse just enough to locate the ROM.
            with open(manifest, "r", encoding="utf-8") as handle:
                # Load metadata.
                meta = json.load(handle)
            # ROM path relative to the pack.
            rom_rel = meta.get("rom", "rom/game.gbc")
            # Absolute ROM path.
            rom_path = os.path.normpath(os.path.join(folder, rom_rel))
            # Skip templates until the ROM is present.
            if not os.path.isfile(rom_path):
                # Keep scaffolds out of the UI dropdown.
                continue
        except Exception:
            # Skip unreadable packs.
            continue
        # Accept this pack id.
        ids.append(name)
    # Return the catalogue.
    return ids


def load_game_pack(game_id: str, games_root: str = DEFAULT_GAMES_ROOT) -> GamePack:
    """Load one game pack by id from the games root."""
    # Absolute pack folder.
    root_dir = os.path.abspath(os.path.join(games_root, game_id))
    # Manifest path.
    manifest_path = os.path.join(root_dir, "game.json")
    # Fail fast when the pack is missing.
    if not os.path.isfile(manifest_path):
        # Clear error for the UI / CLI.
        raise FileNotFoundError(f"Game pack not found: {manifest_path}")
    # Parse the manifest.
    with open(manifest_path, "r", encoding="utf-8") as handle:
        # Load JSON metadata.
        meta = json.load(handle)
    # Prefer explicit id from JSON, else folder name.
    resolved_id = str(meta.get("id", game_id))
    # Display name for banners.
    name = str(meta.get("name", resolved_id))
    # ROM path relative to the pack root.
    rom_rel = meta.get("rom", "rom/game.gbc")
    # Absolute ROM path.
    rom_path = os.path.normpath(os.path.join(root_dir, rom_rel))
    # States directory relative to the pack.
    states_rel = meta.get("states_dir", "states")
    # Absolute states directory.
    states_dir = os.path.normpath(os.path.join(root_dir, states_rel))
    # Memory map filename.
    memory_rel = meta.get("memory_file", "memory.json")
    # Absolute memory path.
    memory_path = os.path.normpath(os.path.join(root_dir, memory_rel))
    # Rewards filename.
    rewards_rel = meta.get("rewards_file", "rewards.json")
    # Absolute rewards path.
    rewards_path = os.path.normpath(os.path.join(root_dir, rewards_rel))
    # Require the ROM to exist before training.
    if not os.path.isfile(rom_path):
        # Point authors at the expected location.
        raise FileNotFoundError(f"ROM not found for game '{resolved_id}': {rom_path}")
    # Require memory + rewards files so packs stay self-contained.
    if not os.path.isfile(memory_path):
        # Missing memory map.
        raise FileNotFoundError(f"memory.json missing for '{resolved_id}': {memory_path}")
    # Require rewards file.
    if not os.path.isfile(rewards_path):
        # Missing rewards.
        raise FileNotFoundError(f"rewards.json missing for '{resolved_id}': {rewards_path}")
    # Build the resolved pack object.
    return GamePack(
        game_id=resolved_id,
        name=name,
        root_dir=root_dir,
        rom_path=rom_path,
        states_dir=states_dir,
        memory_path=memory_path,
        rewards_path=rewards_path,
        meta=meta,
    )


def default_game_id(games_root: str = DEFAULT_GAMES_ROOT) -> Optional[str]:
    """Return the first available game id, preferring mario_deluxe."""
    # Catalogue of installed packs.
    ids = list_game_ids(games_root)
    # Prefer the migrated Mario pack when present.
    if "mario_deluxe" in ids:
        # Historical default game.
        return "mario_deluxe"
    # Otherwise take the first alphabetical pack.
    return ids[0] if ids else None
