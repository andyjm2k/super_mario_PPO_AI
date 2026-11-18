"""Unit tests for the MAME boot-script session."""

import os
import threading
import time
import unittest

from pdt.emulator.mame_console import (
    MameConsoleSession,
    build_boot_script,
    decode_rgb_hex,
    parse_pdt_line,
    write_button_file,
)


class TestPdtProtocol(unittest.TestCase):
    """The button file and PDT line are the contract with the Lua script."""

    def test_button_file_and_parser(self):
        # A session directory without launching MAME.
        session = MameConsoleSession("mame", ".", "sfiii3n", [(0x10, 2), (0x20, 1)], False, start=False)
        # Hold one punch.
        write_button_file(session._buttons, [(":INPUTS", "P1 Jab Punch")])
        # The script must point at that directory and both watches.
        script = build_boot_script(session._dir, session._watches)
        # Button directory is embedded with forward slashes.
        self.assertIn(session._dir.replace("\\", "/").rstrip("/") + "/", script)
        # Health is a 16-bit read.
        self.assertIn("read_u16(16)", script)
        # The flag is an 8-bit read.
        self.assertIn("read_u8(32)", script)
        # A real sample line.
        parsed = parse_pdt_line("PDT 384 224 4 160 1\n", session._watches)
        # Width, height, frame, then the two watches.
        self.assertEqual(parsed[0], 384)
        self.assertEqual(parsed[2], 4)
        self.assertEqual(parsed[3][0x10], 160)
        self.assertEqual(parsed[3][0x20], 1)
        # Log lines are ignored.
        self.assertIsNone(parse_pdt_line("Average speed\n", session._watches))
        # One blue pixel, stored by MAME as B, G, R.
        blue = "ff0000"
        # Pad the token so the parser treats it as a picture.
        picture = blue * 40
        # Width and height are 1, with one RAM sample before the picture.
        parsed_pic = parse_pdt_line("PDT 1 1 8 42 " + picture, [(0x10, 2)])
        # The RAM sample stays in place.
        self.assertEqual(parsed_pic[3][0x10], 42)
        # The first pixel is blue in RGB.
        self.assertEqual(decode_rgb_hex(blue, 1, 1)[0, 0].tolist(), [0, 0, 255])
        # Shut the fake session down.
        session.close()

    def test_step_waits_for_a_newer_frame(self):
        # No MAME process.
        session = MameConsoleSession("mame", ".", "sfiii3n", [(0x10, 2)], False, start=False)
        # A process object the waiter can poll and the closer can stop.
        session._proc = type(
            "Proc",
            (),
            {
                "poll": lambda self: None,
                "terminate": lambda self: None,
                "wait": lambda self, timeout=None: 0,
                "kill": lambda self: None,
            },
        )()

        # A reply that is already waiting must be kept.
        session._queue.put("PDT 16 8 1 144\n")
        # Ask for one frame. No second sample will arrive.
        frame, values = session.step([(":INPUTS", "P1 Up")])
        # The queued sample is the one returned.
        self.assertEqual(values[0x10], 144)
        # Width and height from that line.
        self.assertEqual(frame.shape, (8, 16, 3))
        # The command that would be sent to MAME.
        self.assertIn(":INPUTS\tP1 Up", session._last_command)
        # Missing log still produces a message.
        self.assertIn("missing", session._failure_text("missing"))
        # Stop and delete the directory.
        session.close()
        # The temp directory is gone.
        self.assertFalse(os.path.isdir(session._dir))


if __name__ == "__main__":
    # Run the protocol tests.
    unittest.main()
