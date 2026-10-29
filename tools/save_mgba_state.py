"""Play a GBA pack on mGBA and save a full ``*.mgba_state`` savestate.

The core is headless. This tool opens its own SDL window and blits
``MgbaBackend.get_rgb_frame()`` so the picture is the live framebuffer.

Controls (same as tools/save_pba_state.py):
  A/S = A/B, Arrows = D-pad, Q/W = L/R, Enter = Start, Backspace = Select
  F5  = save ``states/<name>.mgba_state``
  Esc = quit (also saves if --save-on-exit)

``.pba_state`` files are not compatible. Re-record after mGBA is installed.

Example:
  python tools/save_mgba_state.py --game street_fighter3 --name versus_start
"""

from __future__ import annotations

import argparse
import ctypes
import os
import sys

import numpy as np
import sdl2
import sdl2.ext

# Allow running from the repo root without installing the package.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Put the repo on sys.path when it is not already there.
if _ROOT not in sys.path:
    # Local imports resolve to this checkout.
    sys.path.insert(0, _ROOT)

from pdt.emulator.factory import create_emulator_backend, normalize_engine_name
from pdt.emulator.mgba_backend import save_mgba_state
from pdt.game_pack import load_game_pack

# Native GBA width.
_GBA_W = 240
# Native GBA height.
_GBA_H = 160
# Integer scale so the window is large enough to play.
_WINDOW_SCALE = 3


def build_key_maps():
    """Return (keydown, keyup) maps from SDL keycodes to pack button names."""
    # Held while the key is down. Names match the Street Fighter action set.
    key_down = {
        sdl2.SDLK_a: "a",
        sdl2.SDLK_s: "b",
        sdl2.SDLK_BACKSPACE: "select",
        sdl2.SDLK_RETURN: "start",
        sdl2.SDLK_RIGHT: "right",
        sdl2.SDLK_LEFT: "left",
        sdl2.SDLK_UP: "up",
        sdl2.SDLK_DOWN: "down",
        sdl2.SDLK_w: "r",
        sdl2.SDLK_q: "l",
    }
    # Releases use the same mapping.
    key_up = dict(key_down)
    # Both maps for the event loop.
    return key_down, key_up


def _parse_args():
    """Build CLI arguments for the mGBA savestate helper."""
    # Argument parser.
    parser = argparse.ArgumentParser(
        description="Play a GBA pack on mGBA and press F5 to save a .mgba_state"
    )
    # Game pack id under games/.
    parser.add_argument("--game", default="street_fighter3", help="Game pack id")
    # Games root folder.
    parser.add_argument("--games-root", default="games", help="Games root directory")
    # Savestate basename without the extension.
    parser.add_argument(
        "--name",
        default="versus_start",
        help="State file name without extension (saved under states/)",
    )
    # 1 = about real-time inside backend.tick; 0 = unlimited.
    parser.add_argument("--speed", type=float, default=1.0, help="Emulation speed")
    # Also save when the window closes.
    parser.add_argument(
        "--save-on-exit",
        action="store_true",
        help="Write a savestate when quitting with Esc / window close",
    )
    # Parsed namespace.
    return parser.parse_args()


def _save(backend, pack, name: str) -> str:
    """Write one full mGBA state into the pack states directory."""
    # Ensure the states folder exists.
    os.makedirs(pack.states_dir, exist_ok=True)
    # Full output path.
    path = os.path.join(pack.states_dir, f"{name}.mgba_state")
    # Dump CPU + PPU + memory via the backend helper.
    saved = save_mgba_state(backend, path)
    # Tell the user where it landed.
    print(f"Saved mGBA state -> {saved}")
    # Absolute path.
    return saved


class _SdlPlayWindow:
    """SDL2 window that blits live RGB frames from the mGBA backend."""

    def __init__(self, title: bytes = b"mGBA (full savestate)"):
        # Window title bytes for SDL.
        self._title = title
        # Filled in __enter__.
        self._window = None
        # Renderer handle.
        self._renderer = None
        # Streaming RGB24 texture at GBA resolution.
        self._texture = None

    def __enter__(self):
        # Video subsystem only; the emulator itself stays headless.
        if sdl2.SDL_InitSubSystem(sdl2.SDL_INIT_VIDEO) < 0:
            # Surface the SDL error string.
            raise RuntimeError(f"SDL_InitSubSystem failed: {sdl2.SDL_GetError().decode()}")
        # Scaled, resizable play window.
        self._window = sdl2.SDL_CreateWindow(
            self._title,
            sdl2.SDL_WINDOWPOS_CENTERED,
            sdl2.SDL_WINDOWPOS_CENTERED,
            _GBA_W * _WINDOW_SCALE,
            _GBA_H * _WINDOW_SCALE,
            sdl2.SDL_WINDOW_RESIZABLE,
        )
        # Fail if the OS refused the window.
        if not self._window:
            # Bubble the SDL error.
            raise RuntimeError(f"SDL_CreateWindow failed: {sdl2.SDL_GetError().decode()}")
        # Accelerated renderer.
        self._renderer = sdl2.SDL_CreateRenderer(
            self._window, -1, sdl2.SDL_RENDERER_ACCELERATED
        )
        # Fail if the renderer could not be created.
        if not self._renderer:
            # Bubble the SDL error.
            raise RuntimeError(f"SDL_CreateRenderer failed: {sdl2.SDL_GetError().decode()}")
        # Keep the GBA aspect ratio when the window is resized.
        sdl2.SDL_RenderSetLogicalSize(self._renderer, _GBA_W, _GBA_H)
        # RGB24 matches get_rgb_frame channel order.
        self._texture = sdl2.SDL_CreateTexture(
            self._renderer,
            sdl2.SDL_PIXELFORMAT_RGB24,
            sdl2.SDL_TEXTUREACCESS_STREAMING,
            _GBA_W,
            _GBA_H,
        )
        # Fail if the texture could not be created.
        if not self._texture:
            # Bubble the SDL error.
            raise RuntimeError(f"SDL_CreateTexture failed: {sdl2.SDL_GetError().decode()}")
        # Show the window.
        sdl2.SDL_ShowWindow(self._window)
        # Return self for ``with`` usage.
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Destroy the texture when it exists.
        if self._texture:
            # Free the streaming texture.
            sdl2.SDL_DestroyTexture(self._texture)
        # Destroy the renderer when it exists.
        if self._renderer:
            # Free the renderer.
            sdl2.SDL_DestroyRenderer(self._renderer)
        # Destroy the window when it exists.
        if self._window:
            # Free the window.
            sdl2.SDL_DestroyWindow(self._window)
        # Shut down only the video subsystem this tool started.
        sdl2.SDL_QuitSubSystem(sdl2.SDL_INIT_VIDEO)

    def render_rgb(self, rgb_frame: np.ndarray) -> None:
        """Blit one RGB uint8 frame (H x W x 3) to the window."""
        # Contiguous buffer for SDL_UpdateTexture.
        frame = np.ascontiguousarray(rgb_frame, dtype=np.uint8)
        # The texture is fixed at GBA resolution.
        if frame.shape != (_GBA_H, _GBA_W, 3):
            # Reject unexpected layouts.
            raise ValueError(f"Expected ({_GBA_H}, {_GBA_W}, 3) RGB frame, got {frame.shape}")
        # Pointer to the first byte.
        ptr = frame.ctypes.data_as(ctypes.c_void_p)
        # Upload pixels (pitch = width * 3).
        sdl2.SDL_UpdateTexture(self._texture, None, ptr, _GBA_W * 3)
        # Clear the previous frame.
        sdl2.SDL_RenderClear(self._renderer)
        # Copy the texture into the logical viewport.
        sdl2.SDL_RenderCopy(self._renderer, self._texture, None, None)
        # Present.
        sdl2.SDL_RenderPresent(self._renderer)


def main():
    """Open an interactive window and save a full mGBA state on F5."""
    # Parse CLI.
    args = _parse_args()
    # Load the selected pack.
    pack = load_game_pack(args.game, games_root=args.games_root)
    # Canonical engine id.
    engine = normalize_engine_name(pack.emulator_engine)
    # This tool writes .mgba_state files only.
    if engine != "mgba":
        # Point the user at the PyBoyAdvance tool for other GBA packs.
        raise SystemExit(
            f"Pack '{pack.game_id}' uses emulator '{engine}'. "
            "This tool only saves .mgba_state for mgba packs. "
            "Use tools/save_pba_state.py for pyboy_advance packs."
        )
    # Boot headless mGBA (raises a build hint if mgba is not installed).
    backend = create_emulator_backend(
        engine=engine,
        rom_path=pack.rom_path,
        bios_path=pack.bios_path,
        skip_bios=pack.skip_bios,
        emulation_speed=args.speed,
        window="null",
    )
    # SDL keycode -> pack button name.
    key_down_map, key_up_map = build_key_maps()
    # Remember whether F5 ran.
    saved_once = False
    # Print the controls once.
    print(
        f"Playing {pack.name} on mGBA\n"
        "  A/S punch/kick  |  Arrows move  |  Q/W shoulders  |  Enter start\n"
        "  F5 = save .mgba_state  |  Esc = quit\n"
        "Old .pba_state files cannot be loaded. Get into a match, then press F5."
    )
    try:
        # Our window; mGBA itself does not open SDL.
        with _SdlPlayWindow() as window:
            # Main interactive loop.
            running = True
            # Continue until Esc or the window closes.
            while running:
                # Apply input before the frame so the core sees the buttons.
                for event in sdl2.ext.get_events():
                    # Window close.
                    if event.type == sdl2.SDL_QUIT:
                        # Stop the loop.
                        running = False
                    # Key down.
                    elif event.type == sdl2.SDL_KEYDOWN:
                        # Ignore auto-repeat so buttons are edge-triggered.
                        if event.key.repeat:
                            # Skip repeats.
                            continue
                        # Raw keycode.
                        key = event.key.keysym.sym
                        # F5 writes states/<name>.mgba_state.
                        if key == sdl2.SDLK_F5:
                            # Save the full core state.
                            _save(backend, pack, args.name)
                            # Remember that a manual save happened.
                            saved_once = True
                        # Esc quits.
                        elif key == sdl2.SDLK_ESCAPE:
                            # Stop the loop.
                            running = False
                        else:
                            # Translate into a pack button name.
                            mapped = key_down_map.get(key)
                            # Forward presses through the backend.
                            if mapped is not None:
                                # Hold the GBA button.
                                backend.press_button(mapped)
                    # Key up.
                    elif event.type == sdl2.SDL_KEYUP:
                        # Raw keycode.
                        key = event.key.keysym.sym
                        # Esc also ends the loop on key-up.
                        if key == sdl2.SDLK_ESCAPE:
                            # Stop the loop.
                            running = False
                        else:
                            # Translate the release.
                            mapped = key_up_map.get(key)
                            # Forward releases through the backend.
                            if mapped is not None:
                                # Release the GBA button.
                                backend.release_button(mapped)
                # Leave the loop without ticking once more after quit.
                if not running:
                    # Skip the frame advance.
                    break
                # Advance one rendered frame (throttled when --speed > 0).
                backend.tick(1)
                # Draw the live framebuffer, not a frozen snapshot.
                window.render_rgb(backend.get_rgb_frame())
            # Optional save on exit.
            if args.save_on_exit:
                # Dump whatever is on screen.
                _save(backend, pack, args.name)
            elif not saved_once:
                # Reminder when F5 was never pressed.
                print(
                    "No F5 save detected. Re-run and press F5 in-match, "
                    "or pass --save-on-exit to dump on quit."
                )
    finally:
        # Drop the core.
        backend.stop()


if __name__ == "__main__":
    # Script entry point.
    main()
