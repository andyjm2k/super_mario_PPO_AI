"""Play a GBA pack and save PDT ``*.pba_state`` snapshots.

Uses only public PyBoyAdvance APIs (``frame`` / ``press_key`` / ``screen.ndarray``).
The Cython Windows wheel hides ``Window.render``, ``emu.ppu``, and ``emu.keypad``.

Controls:
  A/S = A/B buttons, Arrows = D-pad, Q/W = L/R, Enter = Start, Backspace = Select
  F5  = save snapshot into the pack states/ folder
  Esc = quit (also saves if --save-on-exit)

Example:
  python tools/save_pba_state.py --game street_fighter3 --name versus_start
"""

from __future__ import annotations

import argparse
import ctypes
import os
import sys

import numpy as np
import sdl2
import sdl2.ext

# Allow running from repo root without installing the package.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    # Prepend repo root for local imports.
    sys.path.insert(0, _ROOT)

from pdt.emulator.factory import create_emulator_backend, normalize_engine_name
from pdt.emulator.pyboy_advance_backend import save_pdt_snapshot
from pdt.game_pack import load_game_pack
from pyboy_advance.constants import Key

# Native GBA resolution (Cython wheels do not export DISPLAY_* constants).
_GBA_W = 240
# Native GBA height.
_GBA_H = 160
# Integer window scale factor for comfortable play.
_WINDOW_SCALE = 3


def build_key_maps():
    """Return (keydown_map, keyup_map) from SDL keycodes to Key enums."""
    # Keys held while pressed (fighting-game friendly).
    key_down = {
        sdl2.SDLK_a: Key.BUTTON_A,
        sdl2.SDLK_s: Key.BUTTON_B,
        sdl2.SDLK_BACKSPACE: Key.BUTTON_SELECT,
        sdl2.SDLK_RETURN: Key.BUTTON_START,
        sdl2.SDLK_RIGHT: Key.DPAD_RIGHT,
        sdl2.SDLK_LEFT: Key.DPAD_LEFT,
        sdl2.SDLK_UP: Key.DPAD_UP,
        sdl2.SDLK_DOWN: Key.DPAD_DOWN,
        sdl2.SDLK_w: Key.SHOULDER_RIGHT,
        sdl2.SDLK_q: Key.SHOULDER_LEFT,
    }
    # Matching release map (same codes).
    key_up = dict(key_down)
    # Return both maps for the event loop.
    return key_down, key_up


def _parse_args():
    """Build CLI arguments for the snapshot helper."""
    # Argument parser.
    parser = argparse.ArgumentParser(
        description="Play a GBA pack and press F5 to save a .pba_state"
    )
    # Game pack id under games/.
    parser.add_argument("--game", default="street_fighter3", help="Game pack id")
    # Games root folder.
    parser.add_argument("--games-root", default="games", help="Games root directory")
    # Snapshot basename (without extension).
    parser.add_argument(
        "--name",
        default="versus_start",
        help="Snapshot file name without extension (saved under states/)",
    )
    # Emulation speed (1 = real-time).
    parser.add_argument("--speed", type=float, default=1.0, help="Emulation speed")
    # Also save when closing the window.
    parser.add_argument(
        "--save-on-exit",
        action="store_true",
        help="Write a snapshot when quitting with Esc / window close",
    )
    # Return parsed namespace.
    return parser.parse_args()


def _save(backend, pack, name: str) -> str:
    """Write one PDT snapshot into the pack states directory."""
    # Ensure states folder exists.
    os.makedirs(pack.states_dir, exist_ok=True)
    # Full output path.
    path = os.path.join(pack.states_dir, f"{name}.pba_state")
    # Dump EWRAM/IWRAM/regs via the backend helper.
    saved = save_pdt_snapshot(backend, path)
    # Announce for the user.
    print(f"Saved snapshot -> {saved}")
    # Return absolute path.
    return saved


class _SdlPlayWindow:
    """Minimal SDL2 window that blits RGB frames from ``screen.ndarray``."""

    def __init__(self, title: bytes = b"PyBoy Advance (PDT snapshot)"):
        # Store window title bytes for SDL.
        self._title = title
        # SDL window handle filled in __enter__.
        self._window = None
        # SDL renderer handle.
        self._renderer = None
        # Streaming RGB24 texture matching GBA resolution.
        self._texture = None

    def __enter__(self):
        # Initialize video (+ gamecontroller for pad support later).
        if sdl2.SDL_InitSubSystem(sdl2.SDL_INIT_VIDEO) < 0:
            # Surface a clear SDL error.
            raise RuntimeError(f"SDL_InitSubSystem failed: {sdl2.SDL_GetError().decode()}")
        # Create a scaled, resizable play window.
        self._window = sdl2.SDL_CreateWindow(
            self._title,
            sdl2.SDL_WINDOWPOS_CENTERED,
            sdl2.SDL_WINDOWPOS_CENTERED,
            _GBA_W * _WINDOW_SCALE,
            _GBA_H * _WINDOW_SCALE,
            sdl2.SDL_WINDOW_RESIZABLE,
        )
        # Fail if the window could not be created.
        if not self._window:
            # Bubble up the SDL error string.
            raise RuntimeError(f"SDL_CreateWindow failed: {sdl2.SDL_GetError().decode()}")
        # Hardware-accelerated renderer.
        self._renderer = sdl2.SDL_CreateRenderer(
            self._window, -1, sdl2.SDL_RENDERER_ACCELERATED
        )
        # Fail if renderer creation failed.
        if not self._renderer:
            # Bubble up the SDL error string.
            raise RuntimeError(f"SDL_CreateRenderer failed: {sdl2.SDL_GetError().decode()}")
        # Keep letterboxed GBA aspect on resize.
        sdl2.SDL_RenderSetLogicalSize(self._renderer, _GBA_W, _GBA_H)
        # RGB24 matches ``screen.ndarray`` channel order.
        self._texture = sdl2.SDL_CreateTexture(
            self._renderer,
            sdl2.SDL_PIXELFORMAT_RGB24,
            sdl2.SDL_TEXTUREACCESS_STREAMING,
            _GBA_W,
            _GBA_H,
        )
        # Fail if texture creation failed.
        if not self._texture:
            # Bubble up the SDL error string.
            raise RuntimeError(f"SDL_CreateTexture failed: {sdl2.SDL_GetError().decode()}")
        # Show the window.
        sdl2.SDL_ShowWindow(self._window)
        # Return self for ``with`` usage.
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Destroy texture when present.
        if self._texture:
            # Free the streaming texture.
            sdl2.SDL_DestroyTexture(self._texture)
        # Destroy renderer when present.
        if self._renderer:
            # Free the renderer.
            sdl2.SDL_DestroyRenderer(self._renderer)
        # Destroy window when present.
        if self._window:
            # Free the window.
            sdl2.SDL_DestroyWindow(self._window)
        # Shut down the video subsystem we started.
        sdl2.SDL_QuitSubSystem(sdl2.SDL_INIT_VIDEO)

    def render_rgb(self, rgb_frame: np.ndarray) -> None:
        """Blit one RGB uint8 frame (H x W x 3) to the window."""
        # Require a contiguous buffer for SDL_UpdateTexture.
        frame = np.ascontiguousarray(rgb_frame, dtype=np.uint8)
        # Validate shape matches the GBA framebuffer.
        if frame.shape != (_GBA_H, _GBA_W, 3):
            # Reject unexpected layouts early.
            raise ValueError(f"Expected ({_GBA_H}, {_GBA_W}, 3) RGB frame, got {frame.shape}")
        # Pointer to the first byte of the RGB buffer.
        ptr = frame.ctypes.data_as(ctypes.c_void_p)
        # Upload pixels (pitch = width * 3 bytes).
        sdl2.SDL_UpdateTexture(self._texture, None, ptr, _GBA_W * 3)
        # Clear previous frame contents.
        sdl2.SDL_RenderClear(self._renderer)
        # Copy the full texture to the logical viewport.
        sdl2.SDL_RenderCopy(self._renderer, self._texture, None, None)
        # Present to the display.
        sdl2.SDL_RenderPresent(self._renderer)


def main():
    """Open an interactive window and save snapshots on F5."""
    # Parse CLI.
    args = _parse_args()
    # Load the selected pack.
    pack = load_game_pack(args.game, games_root=args.games_root)
    # Only GBA / PyBoyAdvance packs support .pba_state.
    engine = normalize_engine_name(pack.emulator_engine)
    # Reject GB packs early.
    if engine != "pyboy_advance":
        # Clear error for Mario-style packs.
        raise SystemExit(
            f"Pack '{pack.game_id}' uses emulator '{engine}'. "
            "This tool only saves .pba_state for pyboy_advance packs."
        )
    # Boot the backend (holds the live PyBoyAdvance instance).
    backend = create_emulator_backend(
        engine=engine,
        rom_path=pack.rom_path,
        bios_path=pack.bios_path,
        skip_bios=pack.skip_bios,
        emulation_speed=args.speed,
        window="SDL2",
    )
    # Underlying emulator used for framing / keypad.
    emu = backend._emu
    # SDL keycode -> Key maps for press/release.
    key_down_map, key_up_map = build_key_maps()
    # Track whether the user saved at least once.
    saved_once = False
    # Print controls once.
    print(
        f"Playing {pack.name}\n"
        "  A/S punch/kick  |  Arrows move  |  Q/W shoulders  |  Enter start\n"
        "  F5 = save .pba_state  |  Esc = quit\n"
        "Get into a match (or training), then press F5."
    )
    try:
        # Open our own SDL window (Cython Window.render is not public).
        with _SdlPlayWindow() as window:
            # Main interactive loop.
            running = True
            while running:
                # Advance one rendered frame.
                emu.frame()
                # Poll SDL events ourselves so F5 is visible.
                for event in sdl2.ext.get_events():
                    # Window close.
                    if event.type == sdl2.SDL_QUIT:
                        # Stop the loop.
                        running = False
                    # Key down.
                    elif event.type == sdl2.SDL_KEYDOWN:
                        # Ignore key-repeat for pad buttons.
                        if event.key.repeat:
                            # Skip auto-repeat events.
                            continue
                        # Raw keycode.
                        key = event.key.keysym.sym
                        # F5 saves a snapshot.
                        if key == sdl2.SDLK_F5:
                            # Write states/<name>.pba_state.
                            _save(backend, pack, args.name)
                            # Remember a manual save happened.
                            saved_once = True
                        # Esc quits.
                        elif key == sdl2.SDLK_ESCAPE:
                            # Stop the loop.
                            running = False
                        else:
                            # Translate into a PyBoyAdvance pad press.
                            mapped = key_down_map.get(key)
                            # Forward presses via the public API.
                            if mapped is not None:
                                # Press the mapped GBA key.
                                emu.press_key(mapped)
                    # Key up.
                    elif event.type == sdl2.SDL_KEYUP:
                        # Raw keycode.
                        key = event.key.keysym.sym
                        # Esc also handled on key-up for consistency.
                        if key == sdl2.SDLK_ESCAPE:
                            # Stop the loop.
                            running = False
                        else:
                            # Translate release.
                            mapped = key_up_map.get(key)
                            # Forward releases via the public API.
                            if mapped is not None:
                                # Release the mapped GBA key.
                                emu.release_key(mapped)
                # Draw the RGB frame from the public screen API.
                window.render_rgb(emu.screen.ndarray)
            # Optional save on exit when requested.
            if args.save_on_exit:
                # Force save on exit.
                _save(backend, pack, args.name)
            elif not saved_once:
                # Reminder if they forgot F5.
                print(
                    "No F5 save detected. Re-run and press F5 in-match, "
                    "or pass --save-on-exit to dump on quit."
                )
    finally:
        # Drop emulator resources.
        backend.stop()


if __name__ == "__main__":
    # Script entry point.
    main()
