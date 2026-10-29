"""Headless mGBA backend (hanzi/libmgba-py) with live frames and full savestates."""

from __future__ import annotations

import os
import time
from typing import Optional

from pdt.emulator import BOOT_STATE
from pdt.emulator.mgba_support import (
    _GBA_FPS,
    _GBA_H,
    _GBA_W,
    _key_index,
    _push_keys,
    _require_mgba,
    _rgb_from_image,
    export_state_bytes,
    import_state_bytes,
)


def _run_one_frame(core) -> None:
    """Advance one rendered frame (run_frame or runFrame)."""
    # Snake_case on libmgba-py.
    runner = getattr(core, "run_frame", None) or getattr(core, "runFrame", None)
    # Required for live video.
    if not callable(runner):
        # Do not silently skip frames.
        raise RuntimeError("mGBA core has no run_frame/runFrame")
    # The core writes the new frame into the buffer from set_video_buffer.
    runner()


def _silence_logs() -> None:
    """Install mGBA's null logger when the binding provides one."""
    # Logging is optional and must not block boot.
    try:
        # hanzi/libmgba-py re-exports silence().
        from mgba.log import silence
        # Drop core/game log spam during training.
        silence()
    except Exception:
        # A build without mgba.log still runs.
        return


def _attach_bios(core, bios_path: str, skip_bios: bool) -> None:
    """Load gba_bios.bin and set useBios / skipBios before reset."""
    # Core config is a dict-like mCoreConfig wrapper.
    config = getattr(core, "config", None)
    # Some doubles have no config object.
    if config is not None:
        # Best-effort: a missing key must not abort boot.
        try:
            # Ask the core to use the loaded BIOS image.
            config["useBios"] = True
            # Skip the BIOS intro when the pack says so.
            config["skipBios"] = bool(skip_bios)
        except Exception:
            # Config layout differs; load_bios below is the required part.
            pass
    # VFS opener ships with the bindings.
    from mgba import vfs
    # Absolute path so the C open is independent of cwd.
    vf = vfs.open_path(os.path.abspath(bios_path), "rb")
    # Some builds only accept a read mode without the binary flag.
    if vf is None:
        # Retry without the binary flag.
        vf = vfs.open_path(os.path.abspath(bios_path), "r")
    # Neither open worked.
    if vf is None:
        # BIOS file was present to Python but mGBA could not open it.
        raise RuntimeError(f"mGBA could not open BIOS file: {bios_path}")
    # Snake_case on libmgba-py, camelCase alias just in case.
    load_bios = getattr(core, "load_bios", None) or getattr(core, "loadBIOS", None)
    # Required for a real GBA boot.
    if not callable(load_bios):
        # Do not boot without a BIOS hook.
        raise RuntimeError("This mgba build has no load_bios()")
    # id 0 is the GBA BIOS slot.
    if load_bios(vf) is False:
        # The file opened but the core rejected it.
        raise RuntimeError(f"mGBA rejected BIOS file: {bios_path}")


def _attach_video(core):
    """Allocate a headless RGB buffer and hand it to the core."""
    # GBA panel unless the core reports another size.
    width, height = _GBA_W, _GBA_H
    # Snake_case on libmgba-py.
    dims_fn = getattr(core, "desired_video_dimensions", None) or getattr(
        core, "desiredVideoDimensions", None
    )
    # Query before reset (the method is not needs_reset).
    if callable(dims_fn):
        # A failure here still leaves the GBA default.
        try:
            # (width, height) in pixels.
            dims = dims_fn()
            # Ignore empty or zero results.
            if dims and int(dims[0]) > 0 and int(dims[1]) > 0:
                # Use the core's panel size.
                width, height = int(dims[0]), int(dims[1])
        except Exception:
            # Stick with 240x160.
            pass
    # Image owns the color_t buffer the core will draw into.
    from mgba.image import Image
    # Headless buffer; no SDL window is created here.
    image = Image(width, height)
    # Snake_case setter used by libmgba-py.
    setter = getattr(core, "set_video_buffer", None) or getattr(core, "setVideoBuffer", None)
    # Without this, run_frame has nowhere to draw.
    if not callable(setter):
        # Fail before reset so the caller sees a clear error.
        raise RuntimeError("This mgba build has no set_video_buffer()")
    # Bind the Python buffer for the lifetime of the core.
    setter(image)
    # Caller keeps the image so get_rgb_frame can read it.
    return image


def _boot_core(rom_path: str, bios_path: str, skip_bios: bool):
    """Load a ROM + BIOS into a headless core and reset it."""
    # Import the core module (already verified by _require_mgba).
    import mgba.core as core_mod
    # Quiet logs before the ROM parser runs.
    _silence_logs()
    # load_path is the documented entry point.
    core = core_mod.load_path(rom_path) if hasattr(core_mod, "load_path") else None
    # A None return means the file was not a ROM this core understands.
    if core is None:
        # Do not continue with an empty core.
        raise RuntimeError(f"mGBA could not load ROM: {rom_path}")
    # BIOS and skipBios must be set before reset.
    _attach_bios(core, bios_path, skip_bios)
    # Video buffer must also exist before the first frame.
    image = _attach_video(core)
    # reset() marks the core runnable and builds the memory map.
    reset = getattr(core, "reset", None)
    # Required by run_frame's needs_reset decorator.
    if not callable(reset):
        # Cannot tick without it.
        raise RuntimeError("mGBA core has no reset()")
    # Boot to the post-BIOS (or BIOS) entry point.
    reset()
    # Live core plus the buffer it renders into.
    return core, image


class _MgbaMemoryView:
    """``memory[address]`` byte view over the GBA bus (EWRAM addresses work)."""

    def __init__(self, core):
        # Live core; reads happen at peek time so savestates stay coherent.
        self._core = core

    def __getitem__(self, address: int) -> int:
        # Reward engine passes absolute GBA addresses such as 0x02000818.
        address = int(address) & 0xFFFFFFFF
        # GBAMemory is created on reset and supports memory[addr].
        memory = getattr(self._core, "memory", None)
        # Prefer the bus / raw view shipped with the bindings.
        if memory is not None:
            # __getitem__ is bus-read on 0.1 and raw_read on 0.2.
            try:
                # One byte from the mapped address.
                return int(memory[address]) & 0xFF
            except Exception:
                # Fall through to an explicit raw_read or core.read8.
                pass
            # 8-bit view with an explicit raw_read helper.
            u8 = getattr(memory, "u8", None)
            # raw_read ignores bus side effects (fine for HP peeks).
            if u8 is not None and hasattr(u8, "raw_read"):
                # Offset is absolute because GBAMemory base is 0.
                return int(u8.raw_read(address)) & 0xFF
        # Scripting-style helpers on alternate bindings.
        for name in ("read8", "read_8"):
            # First callable wins.
            reader = getattr(self._core, name, None)
            # Skip missing names.
            if callable(reader):
                # Bus read of one byte.
                return int(reader(address)) & 0xFF
        # No reader could serve this address.
        raise KeyError(f"mGBA cannot read address {address:#010x}")


class MgbaBackend:
    """Headless mGBA adapter: live 240x160 frames and full savestates."""

    def __init__(
        self,
        rom_path: str,
        bios_path: Optional[str] = None,
        skip_bios: bool = True,
        emulation_speed: float = 0,
        window: str = "null",
    ):
        # Fail on missing files before the optional import, so tests stay clear.
        if not rom_path or not os.path.isfile(rom_path):
            # ROM path comes from the game pack.
            raise FileNotFoundError(f"mGBA ROM not found: {rom_path}")
        # A real BIOS is required (same file the PyBoyAdvance pack already uses).
        if not bios_path or not os.path.isfile(bios_path):
            # Point pack authors at game.json "bios".
            raise FileNotFoundError(
                "mGBA requires a GBA BIOS file. Set game.json 'bios' "
                "(Street Fighter uses rom/gba_bios.bin)."
            )
        # Import after path checks so a missing BIOS is not reported as a missing module.
        _require_mgba()
        # Remember the window string for API symmetry (this backend is headless).
        self.window = window
        # 0 = unlimited, used by training; >0 throttles tick() for the save tool.
        self._speed = float(emulation_speed)
        # Held GBA keypad bitmask (active-high, mGBA set_keys raw form).
        self._held = 0
        # Absolute ROM path for debugging.
        self.rom_path = os.path.abspath(rom_path)
        # Boot ROM + BIOS and bind the live framebuffer.
        self._core, self._image = _boot_core(self.rom_path, bios_path, bool(skip_bios))
        # Dict-like memory view for the reward engine.
        self.memory = _MgbaMemoryView(self._core)

    def _require_core(self):
        """Return the live core or raise if stop() already ran."""
        # stop() drops the core.
        if self._core is None:
            # Later calls should not look like a silent no-op.
            raise RuntimeError("mGBA emulator is stopped")
        # Live core.
        return self._core

    def tick(self, count: int = 1) -> None:
        """Advance ``count`` rendered frames so the live framebuffer updates."""
        # At least one frame, matching the other backends.
        frames = max(1, int(count))
        # Core that will draw into self._image.
        core = self._require_core()
        # One run_frame per GBA frame (the buffer is updated in place).
        for _ in range(frames):
            # Render the next frame.
            _run_one_frame(core)
        # Training passes speed 0 and must not sleep.
        if self._speed > 0:
            # Approximate real-time (speed 1) or a faster multiplier.
            time.sleep(frames / (_GBA_FPS * float(self._speed)))

    def press_button(self, name: str) -> None:
        """Hold a pack button (a, b, l, r, d-pad, start, select)."""
        # Resolve to a bit index, then OR it into the held mask.
        self._held |= 1 << _key_index(self._require_core(), name)
        # Push the whole mask so simultaneous buttons stay held.
        _push_keys(self._core, self._held)

    def release_button(self, name: str) -> None:
        """Release one pack button."""
        # Clear that bit and keep the others.
        self._held &= ~(1 << _key_index(self._require_core(), name))
        # Update the core keypad.
        _push_keys(self._core, self._held)

    def release_all(self) -> None:
        """Release every GBA button."""
        # Drop the local mask first.
        self._held = 0
        # Tell the core nothing is held.
        _push_keys(self._require_core(), 0)

    def get_rgb_frame(self):
        """Return the current live RGB frame (160, 240, 3) uint8."""
        # stop() clears the image.
        if self._image is None:
            # No frozen fallback: the core is gone.
            raise RuntimeError("mGBA emulator is stopped")
        # Copy the buffer the last tick() rendered into.
        return _rgb_from_image(self._image)

    def set_emulation_speed(self, speed: float) -> None:
        """Set the tick throttle (0 = unlimited)."""
        # Stored and applied on the next tick.
        self._speed = float(speed)

    def load_state_file(self, path: str) -> None:
        """Load an ``.mgba_state`` blob, or no-op for BOOT_STATE."""
        # Boot sentinel: leave the core at the post-reset entry point.
        if path == BOOT_STATE:
            # Episodes may still boot when no .mgba_state exists yet.
            return
        # Need a live core to receive the blob.
        core = self._require_core()
        # PyBoyAdvance snapshots are not mGBA savestates.
        if str(path).lower().endswith(".pba_state"):
            # Tell the user to re-record with the mGBA tool.
            raise RuntimeError(
                f"Cannot load PyBoyAdvance snapshot '{path}' in mGBA. "
                ".pba_state only restores EWRAM/IWRAM plus a frozen preview. "
                "Re-save a full state with: "
                "python tools/save_mgba_state.py --game street_fighter3 --name versus_start"
            )
        # The file must exist.
        if not os.path.isfile(path):
            # Clear missing-path error.
            raise FileNotFoundError(f"mGBA state not found: {path}")
        # Read the raw blob written by save_mgba_state.
        with open(path, "rb") as handle:
            # Entire savestate (named mGBA state or raw core state).
            blob = handle.read()
        # Restore CPU, PPU, and memory. The live video buffer updates on the next tick.
        import_state_bytes(core, blob)

    def stop(self) -> None:
        """Drop the core. libmgba frees itself when the Core is garbage-collected."""
        # Release the Python reference to the native core.
        self._core = None
        # Drop the framebuffer too so get_rgb_frame cannot return a stale image.
        self._image = None


def save_mgba_state(backend: MgbaBackend, path: str) -> str:
    """Write a full mGBA savestate blob (``.mgba_state``) from a live backend."""
    # Cannot dump a stopped core.
    core = backend._require_core()
    # Bytes from save_state() or save_raw_state().
    blob = export_state_bytes(core)
    # Parent folder is the pack states directory.
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    # Raw blob, not a pickle, so load_state can read it back.
    with open(path, "wb") as handle:
        # Write the savestate bytes.
        handle.write(blob)
    # Absolute path for the tool's log line.
    return os.path.abspath(path)
