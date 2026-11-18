"""Drive stock MAME from a boot script.

The interactive Lua console does not accept piped commands, so each
session writes a script that MAME runs after the board boots. Every
frame the script reads a button file and prints one RAM sample.
"""

from __future__ import annotations

import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from pdt.emulator.mame_frame import bgra_to_rgb, decode_rgb_hex, read_exact
from pdt.emulator.mame_script import build_boot_script

# One held control: MAME port tag, then the field name.
Action = Tuple[str, str]
# One watched word: address, then width in bytes.
Watch = Tuple[int, int]


def write_text_file(path: str, text: str) -> None:
    """Replace a small text file, retrying while a reader still has it open."""
    # Sibling temp name so readers never see a half-written file.
    temporary = path + ".tmp"
    # The full replacement body.
    with open(temporary, "w", encoding="ascii", newline="\n") as handle:
        # Exact bytes the reader will see.
        handle.write(text)
    # Windows refuses the replace while Lua's io.open still holds the destination.
    for _attempt in range(100):
        try:
            # Atomic replace once the reader has closed it.
            os.replace(temporary, path)
            # The new text is in place.
            return
        except PermissionError:
            # The frame callback still has the previous file open.
            time.sleep(0.002)
    # Surface the lock if it never cleared.
    os.replace(temporary, path)


def write_button_file(path: str, actions: Sequence[Action]) -> None:
    """Replace one side of the button buffer."""
    # One port/field pair per line.
    lines = [f"{port}\t{field}\n" for port, field in actions]
    # Same retrying replace as the side marker.
    write_text_file(path, "".join(lines))


def queued_text(item):
    """Return (text, screen bytes) from a queue item."""
    # Pipe closed.
    if item is None:
        # No text and no picture.
        return None, None
    # Reader tuples carry an optional screen dump.
    if isinstance(item, tuple):
        # Header, then bytes.
        return item[0], item[1] if len(item) > 1 else None
    # Unit tests still queue a bare string.
    return item, None


def parse_pdt_line(line: str, watches: Sequence[Watch]):
    """Parse one `PDT width height frame samples... [picture]` line."""
    # Ignore MAME's own log lines.
    if not line.startswith("PDT "):
        # Not a sample.
        return None
    # Split on whitespace.
    bits = line.split()
    # A long hex token at the end is an old picture payload.
    picture = None
    # The word pic means the raw bitmap was written beside the script.
    if len(bits) > 4 and bits[-1] == "pic":
        # Peel the flag so the RAM samples stay aligned.
        picture = "pic"
        # Header plus samples remain.
        bits = bits[:-1]
    # Picture tokens are thousands of hex characters.
    elif len(bits) > 4 and len(bits[-1]) > 64:
        # Peel it off so the RAM samples stay aligned.
        picture = bits[-1]
        # Header plus samples remain.
        bits = bits[:-1]
    # Need width, height, frame, then one sample per watch.
    if len(bits) < 4 + len(watches):
        # Short line.
        return None
    # Picture width.
    width = int(bits[1])
    # Picture height.
    height = int(bits[2])
    # Monotonic frame counter.
    frame_no = int(bits[3])
    # RAM values in watch order.
    values: Dict[int, int] = {}
    # Pair each watch with its sample.
    for index, (address, _width) in enumerate(watches):
        # Samples begin after the three header numbers.
        values[int(address)] = int(bits[4 + index])
    # Parsed sample. Picture is BGR hex, or None when the line has no image.
    return width, height, frame_no, values, picture


class MameConsoleSession:
    """One MAME process controlled by a button file and a boot script."""

    def __init__(self, binary, roms_dir, game_id, watches, render, start: bool = True):
        # Watches in the order the script prints them.
        self._watches = list(watches)
        # Bound the queue. Stdin lockstep stops the board running ahead.
        self._queue: queue.Queue = queue.Queue()
        # Highest frame number already handed to the caller.
        self._last_frame = 0
        # Private directory for the script, buttons, and error log.
        self._dir = tempfile.mkdtemp(prefix="pdt-mame-")
        # Kept so unit tests can still round-trip a button file.
        self._buttons = os.path.join(self._dir, "buttons-0.txt")
        # Last command sent down the pipe. Tests assert on this, not on disk.
        self._last_command = ""
        # Boot script path passed to MAME.
        self._script = os.path.join(self._dir, "boot.lua")
        # Stderr capture so a Lua error is visible when a step times out.
        self._err_path = os.path.join(self._dir, "mame_err.txt")
        # No process until launch.
        self._proc = None
        # Stderr handle owned by this session.
        self._err_handle = None
        # Both sides start empty so the first frame has a file to open.
        write_button_file(os.path.join(self._dir, "buttons-0.txt"), [])
        # The latest command file. Tests read this path.
        self._buttons = os.path.join(self._dir, "buttons-0.txt")
        # Write the frame callback before MAME starts.
        with open(self._script, "w", encoding="ascii", newline="\n") as handle:
            # The boot script.
            handle.write(build_boot_script(self._dir, self._watches))
        # Tests inject no process and stop here.
        if not start:
            # Files exist for the unit test.
            return
        # Refuse to boot a set MAME cannot load.
        self._verify(binary, roms_dir, game_id)
        # Start the board.
        self._launch(binary, roms_dir, game_id, render)

    def step(self, actions: Sequence[Action], capture: bool = True, count: int = 1):
        """Hold these buttons and return the next frame's RAM sample."""
        # One command runs every requested frame inside MAME.
        self._send(actions, capture, count)
        # The reply may already be queued. Do not discard it.
        width, height, frame_no, values, picture, blob = self._wait_frame(self._last_frame)
        # Remember it so the next step waits for something newer.
        self._last_frame = frame_no
        # No bitmap on the fast frames.
        frame = None
        # Raw screen bytes came after the header.
        if blob:
            # Shrink BGRA to the policy image.
            frame = bgra_to_rgb(blob, width, height)
        # Older hex payload, kept for tests.
        elif isinstance(picture, str) and picture != "pic":
            # Decode the downscaled BGR hex.
            frame = decode_rgb_hex(picture, width, height)
        # A requested picture that never arrived still has a shape.
        elif capture:
            # Black stand-in of the reported size.
            frame = decode_rgb_hex(None, width, height)
        # Picture plus memory. frame is None when this tick did not capture.
        return frame, values

    def run_command(self, op: str, path: str, timeout: float = 30.0) -> None:
        """Ask the running board to save or load a snapshot, then wait."""
        # Result from the previous command must not be reused.
        result_path = os.path.join(self._dir, "result.txt")
        # Drop a stale result if one is still on disk.
        if os.path.isfile(result_path):
            # The next write is the only one that counts.
            os.remove(result_path)
        # Forward slashes so the Lua string is a normal path.
        target = os.path.abspath(path).replace("\\", "/")
        # The frame callback reads this file only for save and load, not every step.
        write_text_file(os.path.join(self._dir, "request.txt"), f"{op}\t{target}\n")
        # Unblock the callback so it can see that request.
        self._send([], False)
        # Absolute deadline.
        deadline = time.time() + float(timeout)
        # Keep pulling frames so MAME is not stuck on a full output pipe.
        while time.time() < deadline:
            # The script writes this when the save or load returns.
            if os.path.isfile(result_path):
                try:
                    # Read the status text. Windows may still be closing the writer's handle.
                    with open(result_path, "r", encoding="utf-8", errors="replace") as handle:
                        # "ok" or a Lua error string.
                        status = handle.read().strip()
                    # One result per command.
                    os.remove(result_path)
                except OSError:
                    # The writer still has the file. Try again next pass.
                    continue
                # An empty file is not a finished result.
                if not status:
                    # Wait for the writer to finish.
                    continue
                # Surface Lua failures.
                if status != "ok":
                    # Include the emulator's message.
                    raise RuntimeError(f"MAME {op} failed: {status}")
                # The snapshot command finished.
                return
            try:
                # The poke's sample must not be left for the next step.
                line = self._queue.get(timeout=0.05)
            except queue.Empty:
                # The board has not printed a line yet.
                continue
            # Header text from the reader.
            text, _blob = queued_text(line)
            # Remember the frame number if this line was a sample.
            parsed = parse_pdt_line(text, self._watches) if text else None
            if parsed is not None:
                # Keep the counter ahead of this discarded reply.
                self._last_frame = max(self._last_frame, parsed[2])
        # The callback never answered.
        raise RuntimeError(f"MAME {op} timed out for {target}")

    def close(self) -> None:
        """Stop MAME and delete the session directory."""
        # Only a launched session has a process.
        if self._proc is not None:
            # Ask the process to leave.
            self._proc.terminate()
            try:
                # Short grace.
                self._proc.wait(timeout=3)
            except Exception:
                # Kill a board that ignored terminate.
                self._proc.kill()
        # Close the stderr file if we opened it.
        if self._err_handle is not None:
            # Release the log handle.
            self._err_handle.close()
            # Do not close it twice.
            self._err_handle = None
        # Remove the script and button files.
        shutil.rmtree(self._dir, ignore_errors=True)

    def _verify(self, binary: str, roms_dir: str, game_id: str) -> None:
        """Raise when MAME rejects the rom set."""
        # Audit before paying for a CPS3 boot.
        checked = subprocess.run(
            [binary, game_id, "-rompath", roms_dir, "-verifyroms"],
            capture_output=True,
            text=True,
        )
        # Non-zero means a missing zip or CHD.
        if checked.returncode != 0:
            # Include MAME's own list of missing files.
            detail = (checked.stdout or checked.stderr or "").strip()
            # Stop before launch.
            raise RuntimeError(f"MAME rejected romset '{game_id}' in {roms_dir}.\n{detail}")

    def _launch(self, binary: str, roms_dir: str, game_id: str, render: bool) -> None:
        """Start MAME with the boot script and a stdout reader."""
        # Headless workers skip the GDI window. screen:pixels() still updates.
        if render:
            # The agents the UI is showing get a real window.
            video_flags = ["-video", "gdi", "-window", "-nomaximize", "-resolution", "384x224", "-nokeepaspect"]
        else:
            # No window. The policy still reads the framebuffer.
            video_flags = ["-video", "none"]
        # One arcade board. Commands arrive on stdin. Logs stay in memory.
        self._err_tail = ""
        self._proc = subprocess.Popen(
            [
                binary,
                game_id,
                "-rompath",
                roms_dir,
                "-autoboot_script",
                self._script,
                "-noreadconfig",
                *video_flags,
                "-sound",
                "none",
                "-nothrottle",
                "-skip_gameinfo",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        # Last slice of stderr, kept in memory so a Lua error is visible without a log file.
        self._err_tail = ""
        # Pull stdout off the pipe so MAME cannot block on a full buffer.
        threading.Thread(target=self._read_stdout, name="pdt-mame-stdout", daemon=True).start()
        # Keep stderr from filling the pipe, but do not write it to disk.
        threading.Thread(target=self._read_stderr, name="pdt-mame-stderr", daemon=True).start()
        # The first frame prints READY and then waits for a command.
        self._wait_for_ready()

    def _read_stdout(self) -> None:
        """Copy MAME stdout into the sample queue."""
        # Binary pipe so a screen dump is not newline-translated.
        stdout = self._proc.stdout
        # Until the process closes the pipe.
        while True:
            # One header line.
            raw = stdout.readline()
            # EOF.
            if not raw:
                # Tell waiters that the pipe ended.
                self._queue.put(None)
                # Done.
                return
            # Header is ASCII.
            text = raw.decode("ascii", "replace")
            # Optional screen bytes named by a trailing #count.
            blob = None
            # Split the header.
            parts = text.split()
            # A picture header ends with #nbytes.
            if parts and parts[-1].startswith("#") and parts[-1][1:].isdigit():
                # Exact payload size.
                blob = read_exact(stdout, int(parts[-1][1:]))
            # Header plus optional pixels.
            self._queue.put((text, blob))

    def _wait_for_ready(self) -> None:
        """Block until the frame callback is waiting for a command."""
        # CPS3 boot can take a while before the first frame.
        deadline = time.time() + 120.0
        # Read until the handshake or the board dies.
        while time.time() < deadline:
            try:
                # One stdout line.
                line = self._queue.get(timeout=0.5)
            except queue.Empty:
                # Process death is a hard error.
                if self._proc is not None and self._proc.poll() is not None:
                    # Include the Lua log.
                    raise RuntimeError(self._failure_text("MAME exited during boot"))
                # Keep waiting.
                continue
            # Header text, ignoring any screen bytes.
            text, _blob = queued_text(line)
            # Stdout closed.
            if text is None:
                # The board stopped before the handshake.
                raise RuntimeError(self._failure_text("MAME closed its output pipe during boot"))
            # The callback prints this immediately before reading stdin.
            if str(text).startswith("READY"):
                # Commands written after this are the ones Lua will read.
                return
        # Boot script never reached the frame loop.
        raise RuntimeError(self._failure_text("MAME did not reach the input loop"))

    def _read_stderr(self) -> None:
        """Keep the tail of MAME's log in memory."""
        # Accumulated text, trimmed so a chatty board cannot grow forever.
        tail = ""
        # The pipe closes when the process exits.
        for raw in self._proc.stderr:
            # Stderr is binary when the pipes are.
            line = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
            # Append and keep the last 2KB.
            tail = (tail + line)[-2000:]
            # Published for failure messages.
            self._err_tail = tail

    def _send(self, actions: Sequence[Action], capture: bool, count: int = 1) -> None:
        """Send one command line to the frame callback."""
        # Port and field pairs, separated so the line stays one record.
        fields = "|".join(f"{port}\t{field}" for port, field in actions)
        # Flag, how many frames to run, then the held controls.
        self._last_command = f"{'1' if capture else '0'}|{max(1, int(count))}|{fields}"
        # Tests construct a session with no process.
        stdin = getattr(self._proc, "stdin", None) if self._proc is not None else None
        if stdin is None:
            # Nothing to wake.
            return
        # The callback is blocked in io.stdin:read.
        stdin.write((self._last_command + "\n").encode("ascii"))
        stdin.flush()

    def _drain_baseline(self) -> int:
        """Return the newest frame already queued, without blocking."""
        # Start from the last frame we already returned.
        baseline = self._last_frame
        # Empty whatever arrived before the button write.
        while True:
            try:
                # Non-blocking take.
                line = self._queue.get_nowait()
            except queue.Empty:
                # The queue is caught up.
                return baseline
            # Header text.
            text, _blob = queued_text(line)
            # The reader hit EOF.
            if text is None:
                # Put it back so the waiter reports the exit.
                self._queue.put(None)
                # Keep the baseline we have.
                return baseline
            # A sample newer than the baseline.
            parsed = parse_pdt_line(text, self._watches)
            # Log lines do not move the baseline.
            if parsed is not None and parsed[2] > baseline:
                # Newest frame already produced.
                baseline = parsed[2]

    def _wait_frame(self, baseline: int):
        """Block until a sample with a frame number above `baseline` arrives."""
        # Later frames only wait for the callback, which is already blocked on stdin.
        timeout = 30.0 if baseline else 120.0
        # Absolute deadline.
        deadline = time.time() + timeout
        # Read until a new frame or the deadline.
        while time.time() < deadline:
            # Remaining time for this wait.
            remaining = deadline - time.time()
            try:
                # One stdout line.
                line = self._queue.get(timeout=max(0.1, min(0.5, remaining)))
            except queue.Empty:
                # Process death is a hard error.
                if self._proc is not None and self._proc.poll() is not None:
                    # Include the Lua/MAME log.
                    raise RuntimeError(self._failure_text("MAME exited before a memory sample"))
                # Keep waiting until the deadline.
                continue
            # Header plus any screen bytes that followed it.
            text, blob = queued_text(line)
            # Stdout closed.
            if text is None:
                # The board stopped.
                raise RuntimeError(self._failure_text("MAME closed its output pipe"))
            # Try to parse a sample.
            parsed = parse_pdt_line(text, self._watches)
            # This line was produced after the button write.
            if parsed is not None and parsed[2] > baseline:
                # Width, height, frame, values, picture token, raw bytes.
                return (*parsed, blob)
        # Boot script never printed.
        raise RuntimeError(self._failure_text("MAME did not return a PDT memory sample"))

    def _failure_text(self, summary: str) -> str:
        """Attach the tail of MAME's log to a timeout message."""
        # In-memory stderr from the live process.
        detail = (getattr(self, "_err_tail", "") or "").strip()
        # No log captured.
        if not detail:
            # Summary alone.
            return summary
        # Last portion, so a long boot log does not flood the traceback.
        return summary + "\n" + detail[-2000:]
