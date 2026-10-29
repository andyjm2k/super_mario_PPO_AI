"""Unit tests for pluggable emulator backends."""

import os
import tempfile
import unittest
from array import array
from unittest.mock import MagicMock, patch

import numpy as np

from pdt.emulator import BOOT_STATE
from pdt.emulator.factory import create_emulator_backend, normalize_engine_name
from pdt.emulator.pyboy_advance_backend import (
    PyBoyAdvanceBackend,
    _AdvanceMemoryView,
    save_pdt_snapshot,
)


class TestNormalizeEngine(unittest.TestCase):
    """Verify engine id aliases."""

    def test_pyboy_aliases(self):
        # Canonical pyboy.
        self.assertEqual(normalize_engine_name("pyboy"), "pyboy")
        # GBC alias.
        self.assertEqual(normalize_engine_name("gbc"), "pyboy")

    def test_advance_aliases(self):
        # Underscore form.
        self.assertEqual(normalize_engine_name("pyboy_advance"), "pyboy_advance")
        # Hyphen / short form.
        self.assertEqual(normalize_engine_name("gba"), "pyboy_advance")

    def test_rejects_unknown(self):
        # Unknown engines raise.
        with self.assertRaises(ValueError):
            # Bad id.
            normalize_engine_name("dolphin")


class TestAdvanceMemoryView(unittest.TestCase):
    """Verify the memory[addr] adapter."""

    def test_getitem_calls_read_8_when_available(self):
        # Fake Memory object with a public read_8 + RAM attrs.
        memory = MagicMock()
        # Return a known byte from the bus read.
        memory.read_8.return_value = 42
        # Provide public RAM so buffer discovery succeeds.
        memory.ewram = array("B", [0] * 8)
        # Provide public IWRAM.
        memory.iwram = array("B", [0] * 8)
        # Build the view.
        view = _AdvanceMemoryView(memory)
        # Index like PyBoy memory.
        value = view[0x02000000]
        # Value should pass through read_8.
        self.assertEqual(value, 42)
        # read_8 should have been called once.
        self.assertTrue(memory.read_8.called)

    def test_getitem_reads_ewram_buffer_without_read_8(self):
        # Fake Memory without read_8 (matches Cython wheels).
        memory = MagicMock(spec=[])
        # Seed EWRAM with a marker at offset 0x818.
        ewram = array("B", [0] * 0x900)
        # Marker byte used by Street Fighter P1 HP style peeks.
        ewram[0x818] = 99
        # Attach public buffers.
        memory.ewram = ewram
        # Attach IWRAM.
        memory.iwram = array("B", [0] * 8)
        # Ensure getattr(read_8) is missing.
        self.assertFalse(hasattr(memory, "read_8"))
        # Build the buffer-backed view.
        view = _AdvanceMemoryView(memory)
        # Peek EWRAM absolute address.
        self.assertEqual(view[0x02000818], 99)


class TestPyBoyAdvanceBackend(unittest.TestCase):
    """Cover GBA backend construction and PDT snapshots without a real ROM."""

    def test_missing_bios_raises(self):
        # Constructing without a BIOS file must fail clearly.
        with self.assertRaises(FileNotFoundError):
            # Missing bios path.
            PyBoyAdvanceBackend(rom_path="missing.gba", bios_path=None)

    def test_save_and_load_pdt_snapshot(self):
        # Backend shell without __init__.
        backend = object.__new__(PyBoyAdvanceBackend)
        # Seed cached EWRAM / IWRAM buffers used by the new helpers.
        backend._ewram = array("B", [1, 2, 3, 4])
        # IWRAM payload.
        backend._iwram = array("B", [9, 8, 7, 6])
        # CPU regs buffer.
        backend._regs = array("I", [0] * 16)
        # No RGB cache yet.
        backend._cached_rgb = None
        # UI freeze off until a snapshot with rgb_frame is loaded.
        backend._use_snapshot_rgb = False
        # PC marker.
        backend._regs[15] = 0x08000100
        # Fake live emu for get_rgb_frame during save.
        fake_emu = MagicMock()
        # Non-black RGB so save captures a preview frame.
        fake_emu.screen.ndarray = np.zeros((160, 240, 3), dtype=np.uint8)
        # Paint one bright pixel so max > 0.
        fake_emu.screen.ndarray[0, 0] = [255, 0, 0]
        # Attach fake emu.
        backend._emu = fake_emu
        # Memory view unused here.
        backend.memory = None
        # Key enum unused.
        backend._Key = MagicMock()
        # Temp snapshot path.
        with tempfile.TemporaryDirectory() as tmp:
            # Snapshot path.
            path = os.path.join(tmp, "start.pba_state")
            # Save.
            saved = save_pdt_snapshot(backend, path)
            # File should exist.
            self.assertTrue(os.path.isfile(saved))
            # Mutate live RAM so restore is observable.
            backend._ewram[:] = array("B", [0, 0, 0, 0])
            # Mutate regs so we can prove default load skips them.
            backend._regs[15] = 0
            # Clear RGB cache before load.
            backend._cached_rgb = None
            # Make live PPU a bright title-like frame (would steal UI before freeze).
            fake_emu.screen.ndarray[:] = 200
            # Skip real warm ticks on the shell (no frame API needed).
            backend._warm_boot_for_snapshot = lambda: None
            # Load through the backend helper (boot-then-apply path).
            backend.load_state_file(saved)
            # EWRAM should be restored.
            self.assertEqual(list(backend._ewram), [1, 2, 3, 4])
            # CPU regs must stay at post-mutate values (default skips regs).
            self.assertEqual(int(backend._regs[15]), 0)
            # Snapshot RGB freeze should be armed.
            self.assertTrue(backend._use_snapshot_rgb)
            # UI frame must be the saved match pixel, not live title gray.
            frame = backend.get_rgb_frame()
            # Cached preview pixel must survive a non-black live PPU.
            self.assertEqual(int(frame[0, 0, 0]), 255)

    def test_boot_state_clears_snapshot_rgb_freeze(self):
        # Backend shell.
        backend = object.__new__(PyBoyAdvanceBackend)
        # No emu needed for boot sentinel.
        backend._emu = MagicMock()
        # Pretend a prior snapshot froze the UI.
        backend._use_snapshot_rgb = True
        # Boot sentinel should not raise.
        backend.load_state_file(BOOT_STATE)
        # Live boot frames should be preferred again.
        self.assertFalse(backend._use_snapshot_rgb)

    def test_boot_state_is_noop(self):
        # Backend shell.
        backend = object.__new__(PyBoyAdvanceBackend)
        # No emu needed for boot sentinel.
        backend._emu = MagicMock()
        # Freeze flag present for the boot path.
        backend._use_snapshot_rgb = False
        # Boot sentinel should not raise.
        backend.load_state_file(BOOT_STATE)

    def test_create_backend_routes_pyboy(self):
        # Patch PyBoyBackend so we do not need a ROM.
        with patch("pdt.emulator.factory.PyBoyBackend") as mock_cls:
            # Return a sentinel.
            mock_cls.return_value = "pyboy-backend"
            # Create via factory.
            result = create_emulator_backend("pyboy", rom_path="x.gbc")
            # Factory should return the mock instance.
            self.assertEqual(result, "pyboy-backend")
            # Constructor should receive the ROM path.
            mock_cls.assert_called_once()


if __name__ == "__main__":
    # Run emulator backend unit tests.
    unittest.main()
