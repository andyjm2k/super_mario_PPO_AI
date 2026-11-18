"""MAME arcade backend: one process, two joysticks, RAM for rewards.

Street Fighter III uses the addresses published by MAMEToolkit. Street
Fighter Alpha 3 uses the CPS2 map in the sfa3 pack. When MAMEToolkit can
run (Linux FIFOs) it is used directly. Otherwise this module drives the ``mame`` binary with a Lua boot script,
which is what works on Windows.
"""

from __future__ import annotations

import os
import shutil
from typing import Dict, List, Optional, Tuple

import numpy as np

from pdt.emulator.mame_console import MameConsoleSession
from pdt.emulator.mame_pads import pad_for
from pdt.memory_bank import parse_address


class MameMemoryView:
    """Byte view filled from MAME watches so the reward engine can index it."""

    def __init__(self):
        # Sparse map of address -> byte.
        self._bytes: Dict[int, int] = {}

    def __getitem__(self, address: int) -> int:
        """Return one unsigned byte, or 0 when that address was not watched."""
        # Reward rules index this like PyBoy memory.
        return int(self._bytes.get(int(address), 0)) & 0xFF

    def write_unit(self, address: int, value: int, width: int) -> None:
        """Store a little-endian 1- or 2-byte sample."""
        # Mask to the width we were given.
        raw = int(value) & (0xFFFF if int(width) == 2 else 0xFF)
        # Low byte at the base address.
        self._bytes[int(address)] = raw & 0xFF
        # High byte for halfwords.
        if int(width) == 2:
            # Next address holds the high byte.
            self._bytes[int(address) + 1] = (raw >> 8) & 0xFF


class FakeMameSession:
    """In-process stand-in used by unit tests. No MAME binary required."""

    def __init__(self):
        # Actions received on the last step, as (port, field) pairs.
        self.actions: List[Tuple[str, str]] = []
        # Address -> integer the next step should report.
        self.values: Dict[int, int] = {}
        # Whether close() ran.
        self.closed = False

    def step(self, actions: List[Tuple[str, str]], capture: bool = True, count: int = 1):
        """Record buttons and return a tiny frame plus the scripted RAM."""
        # Remember what the agent pressed.
        self.actions = list(actions)
        # Fast frames do not replace the picture.
        frame = None
        # The last tick of a step is the one the policy sees.
        if capture:
            # 8x8 frame so observation code has a real image.
            frame = np.zeros((8, 8, 3), dtype=np.uint8)
            # A non-zero pixel so tests can see the frame updated.
            frame[0, 0, 0] = 255
        # Copy so later mutations do not change the returned sample.
        return frame, dict(self.values)

    def close(self) -> None:
        """Mark the fake session stopped."""
        # Tests assert this.
        self.closed = True


def _watch_list(memory_bank) -> List[Tuple[int, int]]:
    """Return (address, width) pairs declared by a memory bank."""
    # One watch per signal.
    watches = []
    # Walk the declarative signals.
    for signal in memory_bank.signals.values():
        # Decoder selects the width.
        decode = str(signal.get("decode", "raw")).lower()
        # Base address.
        address = parse_address(signal["address"])
        # Halfword health bars.
        width = 2 if decode.startswith("s16") or decode.startswith("u16") else 1
        # Keep insertion order stable.
        watches.append((address, width))
    # The session reads these every step.
    return watches


class MameBackend:
    """Arcade backend with the same surface as the handheld backends."""

    def __init__(
        self,
        rom_path: str,
        window: str = "null",
        emulation_speed: float = 0,
        session=None,
        game_id: str = "sfiii3n",
    ):
        # Roms directory or a zip path. The parent folder is the rompath.
        self.rom_path = rom_path
        # "null" hides the MAME window during training.
        self._window = window
        # 0 = as fast as the cores run.
        self.emulation_speed = float(emulation_speed)
        # MAME softlist / driver name.
        self.game_id = game_id
        # Injected session (tests) or opened on the first tick.
        self._session = session
        # Byte view the reward engine indexes.
        self.memory = MameMemoryView()
        # (address, width) watches. Empty until bind_memory_bank.
        self._watches: List[Tuple[int, int]] = []
        # Buttons currently held, as (port, field).
        self._held: List[Tuple[str, str]] = []
        # Insert-coin sequence from game.json.
        self._boot_actions: List[dict] = []
        # Last RGB frame.
        self._frame = np.zeros((224, 384, 3), dtype=np.uint8)

    def bind_memory_bank(self, memory_bank) -> None:
        """Register the pack's RAM watches."""
        # Replace any previous list.
        self._watches = _watch_list(memory_bank)

    def set_boot_actions(self, actions: List[dict]) -> None:
        """Store the insert-coin / start sequence."""
        # Copied so later JSON edits do not surprise a running env.
        self._boot_actions = list(actions or [])

    def _ensure_session(self):
        """Open MAME on first use."""
        # Tests inject a session.
        if self._session is not None:
            # Already open.
            return
        # Real MAME or MAMEToolkit.
        self._session = open_mame_session(
            roms_dir=self.rom_path if os.path.isdir(self.rom_path) else os.path.dirname(self.rom_path),
            game_id=self.game_id,
            watches=self._watches,
            render=self._window not in ("null", "headless", "none", ""),
        )

    def press_button(self, name: str) -> None:
        """Hold a player-1 button. Kept for the handheld-style call path."""
        # Single-pad callers mean player 1.
        self.press_player(1, name)

    def press_player(self, player: int, name: str) -> None:
        """Hold one button on player 1 or player 2."""
        # Look up the MAME field for this driver.
        field = pad_for(self.game_id).get(int(player), {}).get(str(name))
        # Unknown names are ignored so a bad pack entry does not crash mid-step.
        if field is None:
            # Nothing to hold.
            return
        # Avoid duplicates while a combo is held.
        if field not in self._held:
            # Remember it until release_all.
            self._held.append(field)

    def release_button(self, name: str) -> None:
        """Release one player-1 button by pack name."""
        # Resolve the player-1 field for this driver.
        field = pad_for(self.game_id).get(1, {}).get(str(name))
        # Drop it if it was held.
        self._held = [item for item in self._held if item != field]

    def release_all(self) -> None:
        """Release every held button on both players."""
        # Next step sends an empty action list.
        self._held = []

    def load_state_file(self, path: str) -> None:
        """Load a MAME snapshot, or insert coins when the path is a boot sentinel."""
        # Open the emulator if this is the first episode.
        self._ensure_session()
        # A real snapshot skips the title-screen coin sequence.
        if path and path != "__boot__" and os.path.isfile(path) and hasattr(self._session, "run_command"):
            # Restore the match the state tool recorded.
            self._session.run_command("load", path)
            # Do not hold the buttons from the previous episode.
            self.release_all()
            # The load is scheduled; a few frames let RAM and the picture catch up.
            self.tick(3)
            # Boot coins are not part of a restored round.
            return
        # Play the insert-coin sequence so a match can start from power-on.
        for spec in self._boot_actions:
            # Which joystick.
            player = int(spec.get("player", 1))
            # Button name from the pad table.
            button = str(spec.get("button", "start"))
            # How many frames to hold it.
            frames = int(spec.get("frames", 2))
            # Press only this boot button.
            self.release_all()
            # Hold it.
            self.press_player(player, button)
            # Step the board.
            self.tick(frames)
        # Clear the coin so it is not held into the match.
        self.release_all()

    def save_state_file(self, path: str) -> str:
        """Write a MAME snapshot the trainer can load as a start state."""
        # The board must be running.
        self._ensure_session()
        # Create the states directory when the tool is saving the first file.
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        # Ask Lua to schedule manager.machine:save. The bytes land on a later frame.
        self._session.run_command("save", path)
        # Absolute path we will return once it is non-empty.
        target = os.path.abspath(path)
        # The scheduled save finishes within a few frames.
        for _ in range(30):
            # Let the board write the file.
            self.tick(1)
            # A real snapshot is many megabytes, never an empty placeholder.
            if os.path.isfile(target) and os.path.getsize(target) > 0:
                # Ready for training to load.
                return target
        # The command returned but no bytes appeared.
        raise RuntimeError(f"MAME save did not write {target}")

    def tick(self, count: int = 1) -> None:
        """Step the arcade board and refresh RAM plus the frame."""
        # At least one frame.
        frames = max(1, int(count))
        # Session must exist.
        self._ensure_session()
        # One command runs every requested frame inside MAME.
        frame, values = self._session.step(list(self._held), capture=True, count=frames)
        # Keep the latest picture when this tick captured one.
        if frame is not None:
            # Store it for the policy and the booth.
            self._frame = np.asarray(frame, dtype=np.uint8)
        # Scatter watched values into the byte view.
        for address, width in self._watches:
            # Missing watches stay at the previous byte.
            if address in values:
                # Store little-endian so s16 rewards can read it.
                self.memory.write_unit(address, values[address], width)

    def get_rgb_frame(self):
        """Return the latest RGB frame."""
        # Copy so callers can hold it.
        return np.array(self._frame, dtype=np.uint8, copy=True)

    def stop(self) -> None:
        """Close the MAME process."""
        # Nothing opened.
        if self._session is not None:
            # Ask the session to exit.
            self._session.close()
            # Drop the handle.
            self._session = None


def find_mame_binary() -> Optional[str]:
    """Return the MAME executable, preferring the copy inside this project."""
    # Explicit override for shells and the trainer UI.
    for key in ("PDT_MAME_BINARY", "MAME_BINARY"):
        # User or process environment.
        configured = os.environ.get(key, "").strip().strip('"')
        # Use it when the file is really there.
        if configured and os.path.isfile(configured):
            # Absolute path.
            return os.path.abspath(configured)
    # Project install: <repo>/mame/mame.exe next to the pdt package.
    bundled = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "mame", "mame.exe")
    )
    # The copy we extracted into the repo.
    if os.path.isfile(bundled):
        # Prefer this over a random mame on PATH.
        return bundled
    # Last resort: whatever is on PATH.
    return shutil.which("mame") or shutil.which("mame.exe")


def open_mame_session(roms_dir: str, game_id: str, watches, render: bool):
    """Open MAMEToolkit when it can run, otherwise the Windows console driver."""
    # Linux FIFO toolkit.
    if os.name != "nt":
        # Import is optional.
        try:
            # The published package.
            from MAMEToolkit.emulator import Address, Emulator
        except ImportError:
            # Fall through to the console driver.
            Emulator = None
        else:
            # Build the address map MAMEToolkit expects.
            address_map = {}
            # One Address per watch.
            for address, width in watches:
                # s16 for health, u8 otherwise.
                kind = "s16" if width == 2 else "u8"
                # Keyed by address so we can scatter results.
                address_map[str(address)] = Address(hex(address), kind)
            # Hand back a thin wrapper.
            return _ToolkitSession(Emulator, roms_dir, game_id, address_map, render)
    # Windows (and Linux without the package): talk to the mame binary.
    binary = find_mame_binary()
    # Nothing to launch.
    if not binary:
        # Tell the user exactly what is missing.
        raise RuntimeError(
            "Street Fighter III needs MAME. Expected "
            f"{os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'mame', 'mame.exe'))} "
            f"or PDT_MAME_BINARY, and the game zip in {roms_dir}."
        )
    # Boot-script driver uses the same watches.
    return MameConsoleSession(binary, roms_dir, game_id, watches, render)


class _ToolkitSession:
    """Adapter over MAMEToolkit.emulator.Emulator."""

    def __init__(self, emulator_cls, roms_dir, game_id, address_map, render):
        # Keep the map so we can read results by address.
        self._address_map = address_map
        # Launch MAME through the toolkit.
        self._emu = emulator_cls(
            "pdt",
            roms_dir,
            game_id,
            address_map,
            frame_ratio=1,
            render=render,
            throttle=False,
            sound=False,
        )

    def step(self, actions):
        """Send MAMEToolkit Action objects and return frame plus RAM."""
        # Local import keeps Windows installs from importing the toolkit.
        from MAMEToolkit.emulator import Action

        # Convert (port, field) into toolkit actions.
        wrapped = [Action(port, field) for port, field in actions]
        # One toolkit step.
        data = self._emu.step(wrapped)
        # Frame may be RGB or ARGB.
        frame = np.asarray(data["frame"])
        # Drop alpha when present.
        if frame.ndim == 3 and frame.shape[2] == 4:
            # Drop the alpha channel.
            frame = frame[:, :, :3]
        # Address string -> int.
        values = {}
        # Scatter every watch.
        for key, value in data.items():
            # Skip the image.
            if key == "frame":
                # Not a memory sample.
                continue
            # Keys are decimal address strings.
            values[int(key)] = int(value)
        # Frame plus RAM.
        return frame, values

    def close(self) -> None:
        """Close the toolkit emulator."""
        # MAMEToolkit registers atexit as well.
        self._emu.close()

