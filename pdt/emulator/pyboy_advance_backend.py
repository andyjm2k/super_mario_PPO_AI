"""PyBoyAdvance (GBA) emulator backend."""

from __future__ import annotations

import os
import pickle
from array import array
from typing import Any, Optional, Tuple

import numpy as np

from pdt.emulator import BOOT_STATE

# PDT-native GBA snapshot format markers (v2 adds an RGB preview frame).
_PBA_FORMAT_V1 = "pdt_pba_v1"
# Current writer format.
_PBA_FORMAT = "pdt_pba_v2"
# Formats accepted by the loader.
_PBA_FORMATS = {_PBA_FORMAT_V1, _PBA_FORMAT}
# Frames to tick before injecting a snapshot so the PPU stays healthy.
_SNAPSHOT_WARM_FRAMES = 8

# Map pack action button names onto PyBoyAdvance Key enums.
_BUTTON_ALIASES = {
    "a": "BUTTON_A",
    "b": "BUTTON_B",
    "select": "BUTTON_SELECT",
    "start": "BUTTON_START",
    "right": "DPAD_RIGHT",
    "left": "DPAD_LEFT",
    "up": "DPAD_UP",
    "down": "DPAD_DOWN",
    "r": "SHOULDER_RIGHT",
    "l": "SHOULDER_LEFT",
    "shoulder_right": "SHOULDER_RIGHT",
    "shoulder_left": "SHOULDER_LEFT",
}


def _get_advance_memory(emu: Any):
    """Return the Memory object (public attr or via cpu on Cython wheels)."""
    # Prefer the documented PyBoyAdvance.memory attribute when present.
    memory = getattr(emu, "memory", None)
    # Fall back to cpu.memory (readonly on the compiled Windows wheel).
    if memory is None:
        # Compiled builds expose Memory only through the CPU.
        memory = emu.cpu.memory
    # Return the live Memory instance.
    return memory


def _memoryview_by_length(owner: Any, length: int):
    """Find a live Cython memoryview buffer by byte/word length."""
    # Prefer a public attribute walk first when pickle is unavailable.
    if not hasattr(owner, "__reduce_cython__"):
        # No Cython pickle helper on pure-Python builds.
        return None
    # Unpack the Cython pickle state tuple.
    state = owner.__reduce_cython__()[2]
    # Scan pickle fields for a buffer of the requested length.
    for item in state:
        # Skip non-sized objects (ints, nested refs, etc.).
        try:
            # Compare buffer length.
            if len(item) == length:
                # Found the matching live buffer.
                return item
        except TypeError:
            # Not a sized buffer; keep scanning.
            continue
    # Nothing matched.
    return None


def _memory_ram_views(memory: Any) -> Tuple[Any, Any]:
    """Return writable (ewram, iwram) buffers for the given Memory object."""
    # Prefer public attributes when the build exposes them to Python.
    ewram = getattr(memory, "ewram", None)
    # Internal work RAM may also be public on pure-Python builds.
    iwram = getattr(memory, "iwram", None)
    # Use them when both are available.
    if ewram is not None and iwram is not None:
        # Public API path.
        return ewram, iwram
    # Import region sizes from the installed package.
    from pyboy_advance.memory.constants import MemoryRegion

    # Locate EWRAM (256 KiB) via Cython pickle state.
    ewram = _memoryview_by_length(memory, int(MemoryRegion.EWRAM_SIZE))
    # Locate IWRAM (32 KiB) the same way.
    iwram = _memoryview_by_length(memory, int(MemoryRegion.IWRAM_SIZE))
    # Both buffers are required for peeking and snapshots.
    if ewram is None or iwram is None:
        # Clear failure when the wheel layout changes.
        raise RuntimeError(
            "PyBoyAdvance Memory buffers are not accessible from Python. "
            "Expected public .ewram/.iwram or Cython pickle views."
        )
    # Return live buffers (mutations are visible to the emulator).
    return ewram, iwram


def _regs_view(cpu: Any):
    """Return the 16-word ARM register file buffer."""
    # Prefer public regs.regs when exposed.
    regs = getattr(cpu.regs, "regs", None)
    # Use it when present.
    if regs is not None:
        # Public path.
        return regs
    # Cython wheels hide cdef regs; find the length-16 buffer.
    regs = _memoryview_by_length(cpu.regs, 16)
    # Require a usable register file for snapshots.
    if regs is None:
        # Fail loudly if the layout changed.
        raise RuntimeError(
            "PyBoyAdvance CPU registers are not accessible from Python."
        )
    # Return the live register buffer.
    return regs


class _AdvanceMemoryView:
    """``memory[addr]`` adapter for PyBoyAdvance (public API or RAM buffers)."""

    def __init__(self, memory):
        # Underlying PyBoyAdvance Memory object.
        self._memory = memory
        # Cache EWRAM/IWRAM for compiled wheels without read_8.
        self._ewram, self._iwram = _memory_ram_views(memory)
        # Import region masks used for address decode.
        from pyboy_advance.memory.constants import MemoryAccess, MemoryRegion

        # Keep MemoryAccess for optional read_8 path.
        self._access = MemoryAccess.NON_SEQUENTIAL
        # EWRAM address mask (0x02000000..0x0203FFFF).
        self._ewram_mask = int(MemoryRegion.EWRAM_MASK)
        # IWRAM address mask (0x03000000..0x03007FFF).
        self._iwram_mask = int(MemoryRegion.IWRAM_MASK)
        # Region id for EWRAM (top byte 0x02).
        self._ewram_region = int(MemoryRegion.EWRAM_REGION)
        # Region id for IWRAM (top byte 0x03).
        self._iwram_region = int(MemoryRegion.IWRAM_REGION)

    def __getitem__(self, address: int) -> int:
        # Normalize to an int GBA bus address.
        address = int(address)
        # Prefer Memory.read_8 when the wheel exposes it (cpdef / pure Python).
        read_8 = getattr(self._memory, "read_8", None)
        # Use the official bus read when available.
        if callable(read_8):
            # Delegate timing-aware reads to the emulator.
            return int(read_8(address, self._access))
        # Compiled wheels only expose RAM via buffers; decode the region.
        region = (address >> 24) & 0xFF
        # External work RAM (most game state lives here).
        if region == self._ewram_region:
            # Index into the 256 KiB EWRAM buffer.
            return int(self._ewram[address & self._ewram_mask])
        # Internal work RAM.
        if region == self._iwram_region:
            # Index into the 32 KiB IWRAM buffer.
            return int(self._iwram[address & self._iwram_mask])
        # Other regions (IO/VRAM/ROM) are not needed for current packs.
        raise KeyError(
            f"Unsupported GBA peek address {address:#010x} "
            "(only EWRAM/IWRAM are readable on this PyBoyAdvance build)"
        )


class PyBoyAdvanceBackend:
    """Adapter around PyBoyAdvance matching the shared emulator protocol."""

    def __init__(
        self,
        rom_path: str,
        bios_path: Optional[str] = None,
        skip_bios: bool = True,
        emulation_speed: float = 0,
        window: str = "null",
    ):
        # Import lazily so GB-only installs do not require the GBA package.
        try:
            from pyboy_advance import PyBoyAdvance
            from pyboy_advance.constants import Key
        except ImportError as exc:
            # Tell the user how to install the optional engine.
            raise ImportError(
                "PyBoyAdvance support requires the pyboy-advance package. "
                "Install with: pip install pyboy-advance"
            ) from exc
        # BIOS is required by PyBoyAdvance for correct boot.
        if not bios_path or not os.path.isfile(bios_path):
            # Point authors at the pack bios field.
            raise FileNotFoundError(
                "PyBoyAdvance requires a GBA BIOS file. "
                "Set game.json 'bios' to a path under the pack "
                "(e.g. rom/gba_bios.bin). Open BIOS: "
                "https://github.com/Nebuleon/ReGBA/blob/master/bios/gba_bios.bin"
            )
        # Store Key enum for button mapping.
        self._Key = Key
        # Window is unused (headless + OpenCV); kept for API symmetry.
        self.window = window
        # Boot the GBA emulator.
        self._emu = PyBoyAdvance(
            rom=rom_path,
            bios=bios_path,
            skip_bios=bool(skip_bios),
            emulation_speed=float(emulation_speed),
        )
        # Resolve Memory (public or cpu.memory on Cython wheels).
        memory = _get_advance_memory(self._emu)
        # Dict-like memory view for the reward / memory bank layer.
        self.memory = _AdvanceMemoryView(memory)
        # Cache RAM buffers used by PDT snapshots.
        self._ewram, self._iwram = self.memory._ewram, self.memory._iwram
        # Cache the ARM register file for optional mid-game restores.
        self._regs = _regs_view(self._emu.cpu)
        # Last known-good RGB frame (PyBoyAdvance has no full savestate API).
        self._cached_rgb = None
        # When True, prefer snapshot RGB over live title/boot frames.
        self._use_snapshot_rgb = False
        # Cache all Key members we may release.
        self._all_keys = [
            Key.BUTTON_A,
            Key.BUTTON_B,
            Key.BUTTON_SELECT,
            Key.BUTTON_START,
            Key.DPAD_RIGHT,
            Key.DPAD_LEFT,
            Key.DPAD_UP,
            Key.DPAD_DOWN,
            Key.SHOULDER_RIGHT,
            Key.SHOULDER_LEFT,
        ]

    def _resolve_key(self, name: str):
        """Map a pack button string onto a Key enum value."""
        # Normalize aliases.
        key_name = _BUTTON_ALIASES.get(str(name).lower().strip(), str(name).upper())
        # Look up on the Key enum.
        if not hasattr(self._Key, key_name):
            # Unknown buttons should fail loudly in pack setup.
            raise ValueError(f"Unknown PyBoyAdvance button '{name}' (resolved '{key_name}')")
        # Return the enum member.
        return getattr(self._Key, key_name)

    def tick(self, count: int = 1) -> None:
        """Advance ``count`` GBA frames (with rendering on the last)."""
        # PyBoyAdvance advances N frames in one call.
        self._emu.frame(count=max(1, int(count)), render=True)

    def press_button(self, name: str) -> None:
        """Press a GBA button by pack action name."""
        # Resolve and press.
        self._emu.press_key(self._resolve_key(name))

    def release_button(self, name: str) -> None:
        """Release a GBA button by pack action name."""
        # Resolve and release.
        self._emu.release_key(self._resolve_key(name))

    def release_all(self) -> None:
        """Release every GBA button we know about."""
        # Walk the full keypad set.
        for key in self._all_keys:
            # Release each key.
            self._emu.release_key(key)

    def get_rgb_frame(self):
        """Return RGB for UI/obs: snapshot preview when frozen, else live screen.

        Incomplete ``.pba_state`` restores leave the live PPU on boot/title
        frames. After a mid-match snapshot load we freeze on the saved
        ``rgb_frame`` so training UI shows the match instead of the title.
        """
        # Prefer the frozen mid-match preview when a snapshot seeded it.
        if self._use_snapshot_rgb and self._cached_rgb is not None:
            # Copy so callers cannot mutate the cache.
            return self._cached_rgb.copy()
        # Live screenshot from the public PyBoyAdvance screen API.
        frame = np.asarray(self._emu.screen.ndarray, dtype=np.uint8)
        # Remember the last non-black frame (optional UI fallback only).
        if frame.size and int(frame.max()) > 0:
            # Cache for rare blank-frame gaps.
            self._cached_rgb = frame.copy()
            # Return the live frame every time it has pixels.
            return frame
        # If the PPU is blank but we cached a preview, use it.
        if self._cached_rgb is not None:
            # Copy so callers cannot mutate the cache.
            return self._cached_rgb.copy()
        # Truly blank (e.g. first frames after BIOS skip).
        return frame

    def set_emulation_speed(self, speed: float) -> None:
        """Set GBA speed multiplier (0 = unlimited)."""
        # Delegate to PyBoyAdvance.
        self._emu.set_emulation_speed(float(speed))

    def load_state_file(self, path: str, restore_cpu_regs: bool = False) -> None:
        """Boot-warm then apply a PDT GBA snapshot, or skip for boot-only starts."""
        # Boot sentinel: leave the emulator at post-BIOS / skip_bios entry.
        if path == BOOT_STATE:
            # Live title/boot frames are intentional for boot-only packs.
            self._use_snapshot_rgb = False
            # Nothing else to restore.
            return
        # Require a file.
        if not os.path.isfile(path):
            # Clear missing-path error.
            raise FileNotFoundError(f"GBA state not found: {path}")
        # Prefer PDT pickle snapshots (PyBoyAdvance has no native save states yet).
        if path.lower().endswith(".pba_state") or path.lower().endswith(".pkl"):
            # Warm the PPU before injecting incomplete RAM (avoids cold crashes).
            self._warm_boot_for_snapshot()
            # Restore our custom format (RAM + RGB; regs optional / unsafe).
            self._load_pdt_snapshot(path, restore_cpu_regs=restore_cpu_regs)
            # Done.
            return
        # Reject opaque PyBoy .state files on the GBA backend.
        raise RuntimeError(
            f"Cannot load '{path}' with PyBoyAdvance. "
            "Use a PDT GBA snapshot (*.pba_state) created by "
            "pdt.emulator.pyboy_advance_backend.save_pdt_snapshot(), "
            "or set allow_boot_start=true for ROM-boot episodes. "
            "Upstream PyBoyAdvance does not yet expose save/load state."
        )

    def _warm_boot_for_snapshot(self) -> None:
        """Tick a few frames so the live PPU is initialized before RAM inject."""
        # Short warm boot keeps screen.ndarray healthy without long title waits.
        self.tick(count=_SNAPSHOT_WARM_FRAMES)

    def _load_pdt_snapshot(self, path: str, restore_cpu_regs: bool = False) -> None:
        """Restore ewram/iwram + rgb_frame after warm boot; skip regs by default.

        Restoring ARM regs into a warm emulator often ACCESS_VIOLATION-crashes
        on the next tick. RAM + frozen rgb_frame is the stable mid-match path.
        """
        # Load the pickle document.
        with open(path, "rb") as handle:
            # Deserialize.
            payload = pickle.load(handle)
        # Validate format marker (v1 and v2).
        if not isinstance(payload, dict) or payload.get("format") not in _PBA_FORMATS:
            # Reject unknown blobs.
            raise RuntimeError(f"Not a PDT GBA snapshot: {path}")
        # Restore external work RAM.
        ewram = payload.get("ewram")
        # Restore internal work RAM.
        iwram = payload.get("iwram")
        # Copy into the live memory arrays when present.
        if ewram is not None:
            # Replace EWRAM contents.
            self._ewram[:] = array("B", ewram)
        # Copy IWRAM.
        if iwram is not None:
            # Replace IWRAM contents.
            self._iwram[:] = array("B", iwram)
        # Optional CPU register restore (unsafe on current PyBoyAdvance wheels).
        regs = payload.get("regs")
        # Only apply when explicitly requested by advanced callers.
        if restore_cpu_regs and regs is not None:
            # Write general-purpose registers.
            for index, value in enumerate(regs):
                # Bound to 16 ARM regs.
                if index < 16:
                    # Set register word on the live buffer.
                    self._regs[index] = int(value) & 0xFFFFFFFF
            # Flush the pipeline after PC changes when the API exists.
            try:
                # Keep fetch/decode coherent.
                self._emu.cpu.flush_pipeline()
            except Exception:
                # Older / Cython builds may not expose flush_pipeline publicly.
                pass
        # RGB preview captured at save time (required for live UI on GBA).
        rgb = payload.get("rgb_frame")
        # Seed the display freeze when the snapshot includes a frame.
        if rgb is not None:
            # Store as contiguous uint8 HxWx3.
            self._cached_rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
            # Keep returning this match frame instead of live title/boot video.
            self._use_snapshot_rgb = True
        else:
            # v1 snapshots have no preview; fall back to live (often blank).
            self._use_snapshot_rgb = False

    def stop(self) -> None:
        """Drop references; PyBoyAdvance has no explicit stop API."""
        # Best-effort: nothing to close for headless mode.
        self._emu = None


def save_pdt_snapshot(backend: PyBoyAdvanceBackend, path: str) -> str:
    """Write a PDT GBA snapshot (*.pba_state) from a live backend."""
    # Require a live emulator.
    if backend._emu is None:
        # Cannot dump a stopped backend.
        raise RuntimeError("Emulator is stopped; cannot save snapshot")
    # Always grab the live PPU (ignore UI freeze used after snapshot loads).
    live = np.asarray(backend._emu.screen.ndarray, dtype=np.uint8)
    # Prefer live pixels; fall back to cache if the PPU is momentarily blank.
    if live.size and int(live.max()) > 0:
        # Contiguous copy for pickle.
        rgb_frame = np.ascontiguousarray(live, dtype=np.uint8)
    elif backend._cached_rgb is not None:
        # Reuse last good frame when live is black.
        rgb_frame = np.ascontiguousarray(backend._cached_rgb, dtype=np.uint8)
    else:
        # Truly blank capture.
        rgb_frame = np.ascontiguousarray(live, dtype=np.uint8)
    # Remember it on the backend too.
    backend._cached_rgb = rgb_frame.copy()
    # Saving is a live capture; do not keep UI frozen on an older preview.
    backend._use_snapshot_rgb = False
    # Build the payload from cached live buffers + RGB preview.
    payload = {
        "format": _PBA_FORMAT,
        "ewram": bytes(backend._ewram),
        "iwram": bytes(backend._iwram),
        "regs": [int(backend._regs[i]) for i in range(16)],
        "rgb_frame": rgb_frame,
    }
    # Ensure parent directory exists.
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    # Write pickle.
    with open(path, "wb") as handle:
        # Dump binary.
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # Return the path written.
    return os.path.abspath(path)
