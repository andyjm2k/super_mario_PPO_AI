"""Unit tests for the headless mGBA backend. No ROM or libmgba required."""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from pdt.emulator import BOOT_STATE
from pdt.emulator.factory import create_emulator_backend, normalize_engine_name
from pdt.emulator.mgba_backend import MgbaBackend, _MgbaMemoryView, save_mgba_state
from pdt.emulator.mgba_support import (
    _require_mgba,
    _rgb_from_555,
    _rgb_from_image,
    export_state_bytes,
    import_state_bytes,
)
from tools.save_mgba_state import build_key_maps

import sdl2


def _rgbx_image(width, height, rgb):
    """Build a fake mGBA image whose buffer is RGBX bytes."""
    # Packed RGBX row, stride == width.
    raw = bytearray(width * height * 4)
    # Paint every pixel from the RGB tuple.
    for index in range(width * height):
        # Channel offset inside the packed buffer.
        base = index * 4
        # Red.
        raw[base] = rgb[0]
        # Green.
        raw[base + 1] = rgb[1]
        # Blue.
        raw[base + 2] = rgb[2]
    # Simple namespace with the attributes _rgb_from_image reads.
    image = MagicMock()
    # Visible width.
    image.width = width
    # Visible height.
    image.height = height
    # No padding.
    image.stride = width
    # Live color buffer.
    image.buffer = raw
    # Return the stand-in image.
    return image


class _FakeCore:
    """Minimal core double covering save, load, frames, keys, and memory."""

    def __init__(self):
        # Last savestate blob.
        self.state = b"STATE-v1"
        # How many frames were rendered.
        self.frames = 0
        # Last keypad mask passed to set_keys.
        self.keys = 0
        # EWRAM-style byte map.
        self.memory = {0x02000818: 0x90}

    def save_state(self):
        # 0.2.0 API: no slot, returns bytes.
        return self.state

    def load_state(self, data):
        # Bytes-oriented test double (not a VFile).
        self.state = bytes(data)
        # Success.
        return True

    def run_frame(self):
        # Count rendered frames.
        self.frames += 1

    def set_keys(self, *args, **kwargs):
        # Record the raw mask.
        self.keys = int(kwargs.get("raw", 0))

    def reset(self):
        # Boot hook used by _boot_core.
        return None


class TestNormalizeMgba(unittest.TestCase):
    """Factory aliases for the mGBA engine."""

    def test_mgba_aliases(self):
        # Canonical id.
        self.assertEqual(normalize_engine_name("mgba"), "mgba")
        # Library alias.
        self.assertEqual(normalize_engine_name("libmgba"), "mgba")
        # Hyphen form folds to the same id.
        self.assertEqual(normalize_engine_name("Lib-MGBA"), "mgba")

    def test_factory_routes_mgba(self):
        # Avoid constructing a real core.
        with patch("pdt.emulator.mgba_backend.MgbaBackend") as mock_cls:
            # Sentinel instance.
            mock_cls.return_value = "mgba-backend"
            # Route through the factory.
            result = create_emulator_backend(
                "libmgba",
                rom_path="x.gba",
                bios_path="bios.bin",
                skip_bios=False,
                emulation_speed=0,
                window="null",
            )
            # Factory returns the constructed backend.
            self.assertEqual(result, "mgba-backend")
            # BIOS and skip flag must be forwarded.
            kwargs = mock_cls.call_args.kwargs
            # ROM path is passed through.
            self.assertEqual(kwargs["rom_path"], "x.gba")
            # BIOS path is passed through.
            self.assertEqual(kwargs["bios_path"], "bios.bin")
            # skip_bios is passed through.
            self.assertFalse(kwargs["skip_bios"])


class TestMgbaImportError(unittest.TestCase):
    """Missing bindings must explain the Windows build."""

    def test_missing_rom_and_bios_before_import(self):
        # Missing ROM fails before mgba is imported.
        with self.assertRaises(FileNotFoundError):
            # No such ROM.
            MgbaBackend(rom_path="missing-rom.gba", bios_path=__file__)
        # Missing BIOS fails the same way.
        with self.assertRaises(FileNotFoundError):
            # This file exists; the BIOS path does not.
            MgbaBackend(rom_path=__file__, bios_path=None)

    def test_missing_mgba_explains_windows_build(self):
        # Real importer, so the message is the one users see.
        import builtins

        # Keep every import except mgba itself.
        real_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            # Fail only the optional emulator package.
            if name == "mgba" or str(name).startswith("mgba."):
                # Same error Python raises when the module is absent.
                raise ImportError("No module named mgba")
            # All other imports stay real.
            return real_import(name, *args, **kwargs)

        # Patch for the duration of the call.
        with patch("builtins.__import__", side_effect=_fake_import):
            # Helper used by MgbaBackend.__init__.
            with self.assertRaises(ImportError) as caught:
                # This must not succeed even if a build lands later.
                _require_mgba()
        # Windows build script is named in the message.
        self.assertIn("build_win64.bat", str(caught.exception))
        # Fork that exposes save_state() -> bytes.
        self.assertIn("libmgba-py", str(caught.exception))
        # pip wheels are called out as insufficient.
        self.assertIn("Windows", str(caught.exception))
        # vcvars64.bat alone does not put MSBuild.exe on PATH.
        self.assertIn("MSBuild", str(caught.exception))
        # The message must name the compiler script that is not enough.
        self.assertIn("vcvars64.bat", str(caught.exception))


class TestMgbaFramesAndMemory(unittest.TestCase):
    """Live framebuffer conversion and bus reads without a real core."""

    def test_rgbx_frame_is_live_copy(self):
        # Solid red RGBX buffer.
        image = _rgbx_image(240, 160, (10, 20, 30))
        # Convert the way the backend does every tick.
        frame = _rgb_from_image(image)
        # Training UI expects HxWx3 at GBA size.
        self.assertEqual(frame.shape, (160, 240, 3))
        # uint8 pixels.
        self.assertEqual(frame.dtype, np.uint8)
        # Channels survived the RGBX unpack.
        self.assertEqual(int(frame[0, 0, 0]), 10)
        # Mutate the source buffer and convert again.
        image.buffer[0] = 200
        # The next read must see the new live pixel, not a cache.
        again = _rgb_from_image(image)
        # Red channel updated.
        self.assertEqual(int(again[0, 0, 0]), 200)

    def test_rgb555_expansion(self):
        # One red-only 555 pixel (low 5 bits = 31).
        colors = np.array([[0x1F]], dtype=np.uint16)
        # Expand to 8-bit.
        rgb = _rgb_from_555(colors)
        # Red is shifted up by 3.
        self.assertEqual(int(rgb[0, 0, 0]), 31 << 3)
        # Green and blue stay dark.
        self.assertEqual(int(rgb[0, 0, 1]), 0)

    def test_memory_dict_and_raw_read(self):
        # Dict-backed memory, like a simplified GBAMemory.
        core = _FakeCore()
        # View under test.
        view = _MgbaMemoryView(core)
        # EWRAM peek used by the Street Fighter reward map.
        self.assertEqual(view[0x02000818], 0x90)
        # Force __getitem__ to fail so raw_read is used.
        class _Boom(dict):
            def __getitem__(self, key):
                # Simulate a bus helper that is not indexable this way.
                raise RuntimeError("no getitem")

        # Replacement memory with a u8 raw_read.
        memory = _Boom()
        # u8 view.
        memory.u8 = MagicMock()
        # Marker byte.
        memory.u8.raw_read.return_value = 77
        # Swap it onto the core.
        core.memory = memory
        # Peek must use raw_read.
        self.assertEqual(view[0x02000818], 77)


class TestMgbaBackendMethods(unittest.TestCase):
    """Cover MgbaBackend methods with a patched boot."""

    def _backend(self):
        # Core double.
        core = _FakeCore()
        # Bright live framebuffer.
        image = _rgbx_image(240, 160, (1, 2, 3))
        # Skip the real importer and the real boot.
        with patch("pdt.emulator.mgba_backend._require_mgba", return_value=object()), patch(
            "pdt.emulator.mgba_backend._boot_core", return_value=(core, image)
        ):
            # Paths must exist; this file is a stand-in.
            backend = MgbaBackend(
                rom_path=__file__,
                bios_path=__file__,
                skip_bios=True,
                emulation_speed=0,
                window="null",
            )
        # Return both so tests can poke the core.
        return backend, core, image

    def test_tick_buttons_frame_and_speed(self):
        # Build the shell.
        backend, core, image = self._backend()
        # Two rendered frames.
        backend.tick(2)
        # Both frames ran.
        self.assertEqual(core.frames, 2)
        # Unlimited speed must not change the stored zero.
        self.assertEqual(backend._speed, 0.0)
        # Change the throttle without sleeping (still zero until set).
        backend.set_emulation_speed(0)
        # Still unlimited.
        self.assertEqual(backend._speed, 0.0)
        # Hold A (bit 0) and L (bit 9).
        backend.press_button("a")
        # A is bit 0.
        self.assertEqual(core.keys & 1, 1)
        # Shoulder L.
        backend.press_button("l")
        # Bit 9 stays with A.
        self.assertEqual(core.keys, (1 << 0) | (1 << 9))
        # Release A only.
        backend.release_button("a")
        # L remains.
        self.assertEqual(core.keys, 1 << 9)
        # Clear the pad.
        backend.release_all()
        # Nothing held.
        self.assertEqual(core.keys, 0)
        # Live frame shape.
        frame = backend.get_rgb_frame()
        # GBA panel.
        self.assertEqual(frame.shape, (160, 240, 3))
        # Paint the buffer and read again.
        image.buffer[2] = 255
        # Blue channel of pixel 0 must update (live, not frozen).
        self.assertEqual(int(backend.get_rgb_frame()[0, 0, 2]), 255)
        # Unknown buttons fail.
        with self.assertRaises(ValueError):
            # Not a GBA key.
            backend.press_button("select_start")

    def test_save_load_roundtrip_and_boot_sentinel(self):
        # Build the shell.
        backend, core, _image = self._backend()
        # Boot sentinel does not touch the blob.
        backend.load_state_file(BOOT_STATE)
        # Original bytes remain.
        self.assertEqual(core.state, b"STATE-v1")
        # Temp folder for the state file.
        with tempfile.TemporaryDirectory() as tmp:
            # Destination path.
            path = os.path.join(tmp, "versus.mgba_state")
            # Mutate the core blob before saving.
            core.state = b"FULL-STATE"
            # Write it.
            saved = save_mgba_state(backend, path)
            # File exists.
            self.assertTrue(os.path.isfile(saved))
            # Change memory so restore is observable.
            core.state = b"other"
            # Load the file back.
            backend.load_state_file(saved)
            # Blob restored.
            self.assertEqual(core.state, b"FULL-STATE")
        # PyBoyAdvance snapshots are rejected.
        with self.assertRaises(RuntimeError):
            # Extension check does not need the file to exist.
            backend.load_state_file("states/versus_start.pba_state")
        # Missing mGBA files fail clearly.
        with self.assertRaises(FileNotFoundError):
            # No such path.
            backend.load_state_file(os.path.join(tempfile.gettempdir(), "nope.mgba_state"))

    def test_stop_blocks_later_calls(self):
        # Build the shell.
        backend, _core, _image = self._backend()
        # Drop the core.
        backend.stop()
        # Ticking a stopped core is an error.
        with self.assertRaises(RuntimeError):
            # Should not render.
            backend.tick(1)
        # Frames are not served from a frozen copy.
        with self.assertRaises(RuntimeError):
            # Image was cleared.
            backend.get_rgb_frame()

    def test_raw_state_fallback_and_vfile_loader(self):
        # Core whose save_state still requires a slot.
        class _SlotCore:
            def save_state(self, slot):
                # 0.1.0 slot API must not be called for a blob.
                raise AssertionError("slot save should be skipped")

            def save_raw_state(self):
                # Raw bytes fallback.
                return b"raw-state"

            def load_raw_state(self, data):
                # Record the restore.
                self.loaded = bytes(data)
                # Success.
                return True

        # Export uses save_raw_state.
        slot_core = _SlotCore()
        # Blob matches the raw saver.
        self.assertEqual(export_state_bytes(slot_core), b"raw-state")
        # Import uses load_raw_state when load_state is absent.
        import_state_bytes(slot_core, b"raw-state")
        # Recorded payload.
        self.assertEqual(slot_core.loaded, b"raw-state")

        # 0.2.0 loader that wants a VFile.
        class _VFileCore:
            def load_state(self, vfile):
                # Remember the object the helper produced.
                self.vfile = vfile
                # Success.
                return True

        # Patch the vfs opener so this test does not import mgba.
        with patch(
            "pdt.emulator.mgba_support._open_state_vfile", return_value="VF"
        ) as opener:
            # Drive the annotated/named vfile path.
            import_state_bytes(_VFileCore(), b"named")
            # Helper received the blob.
            opener.assert_called_once_with(b"named")


class TestSaveMgbaKeyMaps(unittest.TestCase):
    """SDL controls match the PyBoyAdvance snapshot tool."""

    def test_face_and_shoulders(self):
        # Build both maps.
        key_down, key_up = build_key_maps()
        # A / S face buttons.
        self.assertEqual(key_down[sdl2.SDLK_a], "a")
        # B is on S.
        self.assertEqual(key_down[sdl2.SDLK_s], "b")
        # Shoulders.
        self.assertEqual(key_down[sdl2.SDLK_q], "l")
        # Right shoulder.
        self.assertEqual(key_down[sdl2.SDLK_w], "r")
        # Start.
        self.assertEqual(key_down[sdl2.SDLK_RETURN], "start")
        # Releases match presses.
        self.assertEqual(key_up, key_down)


class TestMgbaSmoke(unittest.TestCase):
    """Boot a few frames only when libmgba and the Street Fighter ROM exist."""

    def test_live_frame_shape_if_installed(self):
        # Skip cleanly when the bindings were not built into this venv.
        try:
            # Import probes the real module.
            import mgba.core  # noqa: F401
        except ImportError:
            # Expected until build_win64.bat has been copied into site-packages.
            self.skipTest("mgba is not installed in this venv")
        # Repo root from this test file.
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        # USA ROM shipped with the pack.
        rom = os.path.join(
            root, "games", "street_fighter3", "rom", "Street Fighter Alpha 3 (USA).gba"
        )
        # BIOS next to the ROM.
        bios = os.path.join(root, "games", "street_fighter3", "rom", "gba_bios.bin")
        # Skip when the ROM is not on disk.
        if not os.path.isfile(rom) or not os.path.isfile(bios):
            # Nothing to boot.
            self.skipTest("Street Fighter ROM or BIOS is missing")
        # Headless boot, unlimited speed.
        backend = MgbaBackend(
            rom_path=rom,
            bios_path=bios,
            skip_bios=True,
            emulation_speed=0,
            window="null",
        )
        try:
            # A few frames so the framebuffer is written.
            backend.tick(3)
            # Live RGB panel.
            frame = backend.get_rgb_frame()
            # Shape the training UI scales from.
            self.assertEqual(frame.shape, (160, 240, 3))
            # uint8.
            self.assertEqual(frame.dtype, np.uint8)
            # Reward address must be readable (value depends on boot).
            value = backend.memory[0x02000818]
            # A single byte.
            self.assertGreaterEqual(int(value), 0)
            # And it fits in a byte.
            self.assertLessEqual(int(value), 255)
            # Temp folder for a real named savestate.
            with tempfile.TemporaryDirectory() as tmp:
                # Destination blob.
                path = os.path.join(tmp, "smoke.mgba_state")
                # CPU + PPU + memory.
                save_mgba_state(backend, path)
                # Named states are tens or hundreds of kilobytes.
                self.assertGreater(os.path.getsize(path), 1000)
                # Restore through the same path training uses.
                backend.load_state_file(path)
                # One frame after the restore still has to be a live panel.
                backend.tick(1)
                # Shape is unchanged.
                self.assertEqual(backend.get_rgb_frame().shape, (160, 240, 3))
        finally:
            # Release the core.
            backend.stop()


if __name__ == "__main__":
    # Run these tests directly.
    unittest.main()
