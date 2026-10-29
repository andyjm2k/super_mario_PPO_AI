"""mGBA binding helpers: savestate blobs, live frames, and keypad bits."""

from __future__ import annotations

import inspect
import os
import tempfile
from typing import Optional

import numpy as np


# Native GBA width used when the core does not report dimensions.
_GBA_W = 240
# Native GBA height.
_GBA_H = 160
# Real-time GBA frame rate used when emulation_speed > 0.
_GBA_FPS = 60.0
# mGBA GBA_KEY_* indices (bit positions, not hardware masks).
_KEY_INDEX = {
    "a": 0,
    "b": 1,
    "select": 2,
    "start": 3,
    "right": 4,
    "left": 5,
    "up": 6,
    "down": 7,
    "r": 8,
    "l": 9,
    "shoulder_right": 8,
    "shoulder_left": 9,
}
# Attribute names on mgba.gba.GBA that override the table above.
_KEY_ATTR = {
    "a": "KEY_A",
    "b": "KEY_B",
    "select": "KEY_SELECT",
    "start": "KEY_START",
    "right": "KEY_RIGHT",
    "left": "KEY_LEFT",
    "up": "KEY_UP",
    "down": "KEY_DOWN",
    "r": "KEY_R",
    "l": "KEY_L",
    "shoulder_right": "KEY_R",
    "shoulder_left": "KEY_L",
}
# Shown when `import mgba` fails. pip has no Windows wheel for this binding.
_MGBA_INSTALL_HELP = (
    "mGBA support requires hanzi/libmgba-py (mGBA 0.10.2, MPL 2.0). "
    "pip install mgba and pip install pygba do not provide Windows wheels. "
    "Windows build: 1) Install Visual Studio 2022 (Desktop development with C++) and Git. "
    "2) git clone https://github.com/hanzi/libmgba-py.git "
    "3) vcvars64.bat does NOT add MSBuild. After "
    "VC\\Auxiliary\\Build\\vcvars64.bat, prepend MSBuild\\Current\\Bin to PATH "
    "(typical: C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\MSBuild\\Current\\Bin) "
    "and confirm `where msbuild` prints MSBuild.exe. "
    "4) Activate this repo venv, cd into the clone, run build_win64.bat. "
    "5) Copy build\\win64\\mgba (the package and its DLLs) into the venv "
    "site-packages directory. See games/street_fighter3/README.md. "
    "Old .pba_state files are incompatible; re-save a .mgba_state after import works."
)


def _require_mgba():
    """Import mgba.core or raise an install error that explains the Windows build."""
    # Lazy import so GB-only installs do not need libmgba.
    try:
        # Public core module (load_path / Core live here).
        import mgba.core as core_mod
    except ImportError as exc:
        # Point at hanzi/libmgba-py instead of the empty PyPI names.
        raise ImportError(f"{_MGBA_INSTALL_HELP} Original import error: {exc}") from exc
    # Return the module so callers can load a ROM.
    return core_mod


def _callable_params(fn):
    """Return signature parameters excluding self, or [] when unknown."""
    # cffi / builtin callables may not expose a Python signature.
    try:
        # Inspect the Python wrapper.
        params = list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):
        # Treat an opaque callable as having no extra required args.
        return []
    # Drop the implicit self of bound methods.
    return [param for param in params if param.name != "self"]


def _requires_slot(fn) -> bool:
    """True when save/load still uses the old slot-number API."""
    # Walk required parameters.
    for param in _callable_params(fn):
        # 0.1.0 save_state(slot) cannot return a byte blob.
        if param.name == "slot" and param.default is inspect.Parameter.empty:
            # Caller must use save_raw_state / load_raw_state instead.
            return True
    # 0.2.0 save_state() takes no slot.
    return False


def _wants_vfile(fn) -> bool:
    """True when load_state expects an mGBA VFile (0.2.0)."""
    # Look at the first real parameter.
    params = _callable_params(fn)
    # No parameters means we cannot prove a VFile is required.
    if not params:
        # Fall through to a bytes attempt.
        return False
    # Parameter name used by hanzi/libmgba-py 0.2.0.
    if params[0].name.lower() in ("vfile", "vf"):
        # Open a memory VFile for the blob.
        return True
    # Annotation `vfile: VFile` also marks the 0.2.0 API.
    note = params[0].annotation
    # Missing annotations are not VFile hints.
    if note is inspect.Parameter.empty:
        # Not annotated.
        return False
    # Match the class name without importing mgba.vfs.
    return "VFile" in str(note)


def _as_bytes(data) -> Optional[bytes]:
    """Coerce a save_state result into bytes, or None when it is not a buffer."""
    # Nothing to save.
    if data is None:
        # Caller tries the next API.
        return None
    # Already a Python buffer (0.2.0 save_state returns bytes).
    if isinstance(data, (bytes, bytearray, memoryview)):
        # Copy so the caller owns the blob.
        return bytes(data)
    # cffi `unsigned char[]` from save_raw_state.
    try:
        # Import only when a real build returned a cdata object.
        from mgba._pylib import ffi
        # Copy the native buffer out of cffi.
        return bytes(ffi.buffer(data))
    except Exception:
        # Not a cffi buffer.
        return None


def export_state_bytes(core) -> bytes:
    """Read a full savestate blob from whatever API this mgba build exposes."""
    # Prefer 0.2.0 save_state() -> bytes (CPU + PPU + memory).
    save = getattr(core, "save_state", None)
    # Skip the 0.1.0 slot saver; it does not return bytes.
    if callable(save) and not _requires_slot(save):
        # Call the no-argument saver.
        data = _as_bytes(save())
        # Use it when the build returned a buffer.
        if data:
            # Full named savestate blob.
            return data
    # Older builds expose the raw core struct instead.
    raw_fn = getattr(core, "save_raw_state", None)
    # Try that path when present.
    if callable(raw_fn):
        # May be bytes or a cffi array.
        data = _as_bytes(raw_fn())
        # Accept a non-empty raw dump.
        if data:
            # Raw state matches load_raw_state.
            return data
    # Neither API produced a blob.
    raise RuntimeError(
        "This mgba build cannot export a savestate blob. "
        "Need Core.save_state() -> bytes (libmgba-py >= 0.2.0) or save_raw_state()."
    )


def _open_state_vfile(blob: bytes):
    """Wrap savestate bytes in a native mGBA VFile for load_state(vfile)."""
    # Local import: tests patch this helper so mgba.vfs is optional.
    from mgba import vfs
    # Unique temp path. The Python BytesIO VFile always reports a full read,
    # including at EOF, and mCoreLoadStateNamed then spins forever.
    fd, path = tempfile.mkstemp(prefix="pdt-mgba-", suffix=".mgba_state")
    try:
        # Write the named savestate blob to disk.
        os.write(fd, bytes(blob))
    finally:
        # Close our handle before mGBA opens the same path.
        os.close(fd)
    # Native reader (VFileOpen), which returns a short read at EOF.
    vf = vfs.open_path(path, "rb")
    # open_path returns None when the OS open fails.
    if vf is None:
        # Don't leave the blob behind.
        os.remove(path)
        # The caller cannot load without a file handle.
        raise RuntimeError("mGBA could not open a temporary savestate file")
    # Remember the path so the load can delete it afterwards.
    vf._pdt_temp_path = path
    # VFile the core will read during load_state.
    return vf


def _release_state_vfile(vf) -> None:
    """Close a native state VFile and delete its temp file."""
    # Test doubles are strings or mocks and have no temp path.
    path = getattr(vf, "_pdt_temp_path", None)
    # Nothing we created.
    if not path:
        # Leave the fake loader's object alone.
        return
    # Close the C file before unlinking it on Windows.
    close = getattr(vf, "close", None)
    # Some objects are not closable.
    if callable(close):
        # A close error must not hide whether the state loaded.
        try:
            # Release the handle mGBA opened.
            close()
        except Exception:
            # Still try to delete the file below.
            pass
    # Remove the temp blob.
    try:
        # Best-effort unlink.
        os.remove(path)
    except OSError:
        # Already gone, or still locked.
        pass


def import_state_bytes(core, blob: bytes) -> None:
    """Restore a full savestate blob into a live core."""
    # Own a plain bytes copy.
    payload = bytes(blob)
    # Empty files are not states.
    if not payload:
        # Fail before touching the core.
        raise RuntimeError("Refusing to load an empty mGBA state")
    # 0.2.0 loader, or a test double named load_state.
    load = getattr(core, "load_state", None)
    # Use it when the build provides the method.
    if callable(load) and not _requires_slot(load):
        # Named-state API consumes a VFile, not raw bytes.
        if _wants_vfile(load):
            # Native temp-file VFile (see _open_state_vfile).
            opened = _open_state_vfile(payload)
            try:
                # Restore CPU, PPU, and memory from that file.
                result = load(opened)
            finally:
                # Drop the temp file once the core has finished reading.
                _release_state_vfile(opened)
        else:
            # Test doubles and any bytes-oriented loader.
            try:
                # Pass the blob straight through.
                result = load(payload)
            except (TypeError, AttributeError):
                # Signature claimed bytes but the body wanted a VFile.
                opened = _open_state_vfile(payload)
                try:
                    # Retry with the native file wrapper.
                    result = load(opened)
                finally:
                    # Drop the temp file.
                    _release_state_vfile(opened)
        # False / 0 means the core rejected the blob (cffi may return 0).
        if result is False or result == 0:
            # Do not pretend the state loaded.
            raise RuntimeError("mGBA load_state rejected the savestate blob")
        # Loaded.
        return
    # Fallback: internal raw state from save_raw_state.
    raw_load = getattr(core, "load_raw_state", None)
    # Present on both 0.1.0 and 0.2.0.
    if callable(raw_load):
        # One call: False / 0 means the buffer was shorter than stateSize.
        raw_result = raw_load(payload)
        # cffi may return 0 instead of Python False.
        if raw_result is False or raw_result == 0:
            # Surface the rejection.
            raise RuntimeError("mGBA load_raw_state rejected the savestate blob")
        # Loaded.
        return
    # No usable loader.
    raise RuntimeError(
        "This mgba build cannot import a savestate blob. "
        "Need Core.load_state(vfile) or load_raw_state()."
    )


def _image_memoryview(image):
    """Return a memoryview of an mgba.image.Image color buffer."""
    # Python buffer or cffi array attached by set_video_buffer.
    buffer = getattr(image, "buffer", None)
    # A missing buffer means video was never attached.
    if buffer is None:
        # Caller turns this into an empty frame error.
        raise RuntimeError("mGBA video image has no buffer")
    # bytearray / bytes used by tests, and some cffi buffers.
    try:
        # Prefer the buffer protocol.
        return memoryview(buffer)
    except TypeError:
        # cdata arrays need ffi.buffer before memoryview.
        pass
    # Real libmgba-py images store a cffi color_t array.
    from mgba._pylib import ffi
    # Expose the native pixels to numpy.
    return memoryview(ffi.buffer(buffer))


def _rgb_from_555(colors: np.ndarray) -> np.ndarray:
    """Expand mGBA 16-bit RGB555 pixels to uint8 RGB."""
    # Work in 16-bit so shifts stay in range.
    colors = colors.astype(np.uint16)
    # Low 5 bits are red (mGBA color_t layout).
    red = ((colors & 0x1F) << 3).astype(np.uint8)
    # Next 5 bits are green.
    green = (((colors >> 5) & 0x1F) << 3).astype(np.uint8)
    # Next 5 bits are blue.
    blue = (((colors >> 10) & 0x1F) << 3).astype(np.uint8)
    # Stack into HxWx3.
    return np.stack([red, green, blue], axis=-1)


def _rgb_from_image(image) -> np.ndarray:
    """Copy the live framebuffer to an RGB uint8 array (never a frozen snapshot)."""
    # Reported width, defaulting to the GBA panel.
    width = int(getattr(image, "width", _GBA_W) or _GBA_W)
    # Reported height.
    height = int(getattr(image, "height", _GBA_H) or _GBA_H)
    # Stride in pixels (may be wider than the visible width).
    stride = int(getattr(image, "stride", width) or width)
    # Live pixels written by the last run_frame.
    raw = np.frombuffer(_image_memoryview(image), dtype=np.uint8)
    # Nothing rendered yet.
    if raw.size == 0 or height <= 0 or width <= 0:
        # Blank frame with the expected rank.
        return np.zeros((max(height, 1), max(width, 1), 3), dtype=np.uint8)
    # Bytes per row, including padding.
    row_bytes = raw.size // height
    # color_t is 4 bytes (XBGR8/RGBX) on 0.10 and 2 bytes on older builds.
    bpp = max(1, row_bytes // max(1, stride))
    # Drop stride padding past the visible width.
    pixels = raw.reshape((height, row_bytes))[:, : width * bpp]
    # HxWxC view of the visible panel.
    pixels = pixels.reshape((height, width, bpp))
    # 32-bit buffer is RGBX in memory (same order as Image.to_pil).
    if bpp >= 3:
        # Drop the unused X/A byte.
        rgb = pixels[:, :, :3]
    elif bpp == 2:
        # Pack the two bytes into RGB555 words.
        words = pixels[:, :, 0].astype(np.uint16) | (pixels[:, :, 1].astype(np.uint16) << 8)
        # Expand to 8-bit RGB.
        rgb = _rgb_from_555(words)
    else:
        # Single-channel fallback, repeated so the UI still gets 3 channels.
        rgb = np.repeat(pixels[:, :, :1], 3, axis=2)
    # Copy so callers cannot alias the live core buffer.
    return np.ascontiguousarray(rgb, dtype=np.uint8)


def _key_index(core, name: str) -> int:
    """Map a pack button name onto an mGBA key bit index."""
    # Normalize pack ids (`L`, `left`, ...).
    key = str(name).lower().strip()
    # Prefer constants compiled into this core (GBA.KEY_*).
    attr = _KEY_ATTR.get(key)
    # Present on mgba.gba.GBA.
    if attr and hasattr(core, attr):
        # Enum value is an index, or occasionally a single-bit mask.
        value = int(getattr(core, attr))
        # A lone high bit means the binding exported a mask instead of an index.
        if value > 9 and value & (value - 1) == 0:
            # Convert the mask back into a bit index for set_keys.
            return int(value.bit_length() - 1)
        # Index form used by Core._keys_to_int.
        return value
    # Built-in table matches mGBA 0.10 GBA_KEY_*.
    if key not in _KEY_INDEX:
        # Unknown pack buttons should fail during setup.
        raise ValueError(f"Unknown mGBA button '{name}'")
    # Default bit index.
    return _KEY_INDEX[key]


def _push_keys(core, mask: int) -> None:
    """Write the held-button bitmask into the core."""
    # 0.2.0 / 0.1.0 snake_case, then a camelCase alias.
    set_keys = getattr(core, "set_keys", None) or getattr(core, "setKeys", None)
    # Preferred path: one raw mask.
    if callable(set_keys):
        # set_keys(raw=mask) is the documented Core API.
        try:
            # Pass the bitmask through the raw keyword.
            set_keys(raw=int(mask))
            # Applied.
            return
        except TypeError:
            # Some wrappers take the integer mask positionally.
            set_keys(int(mask))
            # Applied.
            return
    # Last resort: clear every key, then add the held indices.
    clear = getattr(core, "clear_keys", None) or getattr(core, "clearKeys", None)
    # Matching adder.
    add = getattr(core, "add_keys", None) or getattr(core, "addKeys", None)
    # Both are required for this fallback.
    if not callable(clear) or not callable(add):
        # Give up with a clear message.
        raise RuntimeError("mGBA core has no set_keys/add_keys API")
    # Release the whole keypad (10 GBA buttons).
    try:
        # Raw mask covering A through L.
        clear(raw=0x3FF)
    except TypeError:
        # Index form.
        clear(*range(10))
    # Bits that are still held.
    held = [bit for bit in range(10) if mask & (1 << bit)]
    # Nothing held after a release_all.
    if not held:
        # Keypad is clear.
        return
    # Press the remaining indices.
    try:
        # Index form used by Core.add_keys.
        add(*held)
    except TypeError:
        # Mask form.
        add(raw=int(mask))

