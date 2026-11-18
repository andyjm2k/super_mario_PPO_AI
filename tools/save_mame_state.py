"""Play a MAME pack and save a ``*.sta`` snapshot for training.

Both fighters are on one board. Reach the round you want, then press F5.
Training loads every ``games/<pack>/states/*.sta`` file as a start state.

Player 1: arrows, Q/W/E punches, A/S/D kicks, 1 coin, Enter start
Player 2: I up, J left, K down, L right, U/O/P punches, N/M/, kicks, 2 coin, Backspace start
F5 saves. Esc quits.

Example:
  python tools/save_mame_state.py --game sfiii3n --name versus_start
"""

from __future__ import annotations

import argparse
import ctypes
import os
import sys
import time

import numpy as np
import sdl2

# Allow running from the repo root without installing the package.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Put the repo on sys.path when it is not already there.
if _ROOT not in sys.path:
    # Local imports resolve to this checkout.
    sys.path.insert(0, _ROOT)

from pdt.emulator.factory import create_emulator_backend, normalize_engine_name
from pdt.game_pack import load_game_pack

# How long one visible frame should last, in seconds.
_FRAME_SECONDS = 1.0 / 60.0


def build_key_maps():
    """Return keycode -> (player, button) for keys that are currently held."""
    # Player 1 on the left of a normal keyboard, player 2 on the right.
    mapping = {
        sdl2.SDLK_UP: (1, "up"),
        sdl2.SDLK_DOWN: (1, "down"),
        sdl2.SDLK_LEFT: (1, "left"),
        sdl2.SDLK_RIGHT: (1, "right"),
        sdl2.SDLK_q: (1, "lp"),
        sdl2.SDLK_w: (1, "mp"),
        sdl2.SDLK_e: (1, "hp"),
        sdl2.SDLK_a: (1, "lk"),
        sdl2.SDLK_s: (1, "mk"),
        sdl2.SDLK_d: (1, "hk"),
        sdl2.SDLK_1: (1, "coin"),
        sdl2.SDLK_RETURN: (1, "start"),
        sdl2.SDLK_i: (2, "up"),
        sdl2.SDLK_k: (2, "down"),
        sdl2.SDLK_j: (2, "left"),
        sdl2.SDLK_l: (2, "right"),
        sdl2.SDLK_u: (2, "lp"),
        sdl2.SDLK_o: (2, "mp"),
        sdl2.SDLK_p: (2, "hp"),
        sdl2.SDLK_n: (2, "lk"),
        sdl2.SDLK_m: (2, "mk"),
        sdl2.SDLK_COMMA: (2, "hk"),
        sdl2.SDLK_2: (2, "coin"),
        sdl2.SDLK_BACKSPACE: (2, "start"),
    }
    # The same table is used for keydown and keyup.
    return mapping


def _parse_args():
    """Build CLI arguments for the MAME snapshot helper."""
    # Argument parser.
    parser = argparse.ArgumentParser(description="Play a MAME pack and press F5 to save a .sta")
    # Game pack id under games/.
    parser.add_argument("--game", default="sfiii3n", help="Game pack id")
    # Games root folder.
    parser.add_argument("--games-root", default="games", help="Games root directory")
    # Snapshot basename without the extension.
    parser.add_argument("--name", default="versus_start", help="State name without extension")
    # Also save when the window closes.
    parser.add_argument("--save-on-exit", action="store_true", help="Save when quitting")
    # Parsed namespace.
    return parser.parse_args()


def _save(backend, pack, name: str) -> str:
    """Write one MAME snapshot into the pack states directory."""
    # Ensure the states folder exists.
    os.makedirs(pack.states_dir, exist_ok=True)
    # Full output path.
    path = os.path.join(pack.states_dir, f"{name}.sta")
    # Ask the running board to write the file.
    saved = backend.save_state_file(path)
    # Tell the user where it landed.
    print(f"Saved MAME state -> {saved}")
    # Absolute path.
    return saved


class _PlayWindow:
    """SDL window that shows the latest MAME frame at its native size."""

    def __init__(self, width: int, height: int):
        # Texture width.
        self._width = int(width)
        # Texture height.
        self._height = int(height)
        # Filled in open().
        self._window = None
        # Renderer.
        self._renderer = None
        # Streaming texture.
        self._texture = None

    def open(self):
        """Create the window."""
        # Video only.
        if sdl2.SDL_InitSubSystem(sdl2.SDL_INIT_VIDEO) < 0:
            # Surface the SDL error.
            raise RuntimeError(f"SDL_InitSubSystem failed: {sdl2.SDL_GetError().decode()}")
        # A window large enough to see a 384x224 arcade screen.
        self._window = sdl2.SDL_CreateWindow(
            b"MAME versus (F5 saves)",
            sdl2.SDL_WINDOWPOS_CENTERED,
            sdl2.SDL_WINDOWPOS_CENTERED,
            self._width * 2,
            self._height * 2,
            sdl2.SDL_WINDOW_RESIZABLE,
        )
        # Accelerated renderer.
        self._renderer = sdl2.SDL_CreateRenderer(self._window, -1, sdl2.SDL_RENDERER_ACCELERATED)
        # Keep the game aspect when the window is resized.
        sdl2.SDL_RenderSetLogicalSize(self._renderer, self._width, self._height)
        # RGB24 matches get_rgb_frame.
        self._texture = sdl2.SDL_CreateTexture(
            self._renderer,
            sdl2.SDL_PIXELFORMAT_RGB24,
            sdl2.SDL_TEXTUREACCESS_STREAMING,
            self._width,
            self._height,
        )
        # Show it.
        sdl2.SDL_ShowWindow(self._window)

    def close(self):
        """Destroy the window."""
        # Texture.
        if self._texture:
            # Free it.
            sdl2.SDL_DestroyTexture(self._texture)
        # Renderer.
        if self._renderer:
            # Free it.
            sdl2.SDL_DestroyRenderer(self._renderer)
        # Window.
        if self._window:
            # Free it.
            sdl2.SDL_DestroyWindow(self._window)
        # Video subsystem.
        sdl2.SDL_QuitSubSystem(sdl2.SDL_INIT_VIDEO)

    def render_rgb(self, frame: np.ndarray) -> None:
        """Blit one RGB frame, resized to the texture when MAME's size differs."""
        # Contiguous RGB.
        image = np.ascontiguousarray(frame, dtype=np.uint8)
        # The texture is fixed. Resize when the sample is another size.
        if image.shape[0] != self._height or image.shape[1] != self._width:
            # Local import keeps the module import light.
            import cv2
            # Match the texture.
            image = cv2.resize(image, (self._width, self._height))
            # Contiguous again after resize.
            image = np.ascontiguousarray(image, dtype=np.uint8)
        # First byte.
        ptr = image.ctypes.data_as(ctypes.c_void_p)
        # Upload.
        sdl2.SDL_UpdateTexture(self._texture, None, ptr, self._width * 3)
        # Clear.
        sdl2.SDL_RenderClear(self._renderer)
        # Draw.
        sdl2.SDL_RenderCopy(self._renderer, self._texture, None, None)
        # Present.
        sdl2.SDL_RenderPresent(self._renderer)


def run(args) -> None:
    """Play until F5 or Esc, saving a .sta the trainer will load."""
    # Selected pack.
    pack = load_game_pack(args.game, games_root=args.games_root)
    # This tool is the arcade saver.
    if normalize_engine_name(pack.emulator_engine) != "mame":
        # Point at the GBA tool otherwise.
        raise SystemExit(
            f"Pack '{pack.game_id}' is not a MAME pack. "
            "Use tools/save_mgba_state.py for mGBA."
        )
    # Boot the board. The backend opens the MAME window that actually draws.
    backend = create_emulator_backend(
        engine="mame",
        rom_path=pack.rom_path,
        window="null",
        emulation_speed=0,
    )
    # Watch the reward addresses so a saved state has a live memory map too.
    if hasattr(backend, "bind_memory_bank"):
        # Pack map.
        backend.bind_memory_bank(pack.load_memory())
    # Romset name.
    if hasattr(backend, "game_id"):
        # sfiii3n, not the folder name alone.
        backend.game_id = str(pack.meta.get("mame_game") or pack.game_id)
    # Insert coins so the player can reach a match before saving.
    if hasattr(backend, "set_boot_actions"):
        # Same sequence training uses for a cold boot.
        backend.set_boot_actions(list(pack.meta.get("boot_actions", [])))
    # Cold boot. The player then moves to the round they want to save.
    backend.load_state_file("__boot__")
    # Key table.
    mapping = build_key_maps()
    # Buttons currently held, as (player, name).
    held = set()
    # Preview window sized to a CPS picture. The first frame sets the real size.
    first = np.asarray(backend.get_rgb_frame())
    # Window.
    window = _PlayWindow(int(first.shape[1]), int(first.shape[0]))
    # Create it.
    window.open()
    # Draw the first picture.
    window.render_rgb(first)
    # Run until Esc or the window closes.
    running = True
    # Event.
    event = sdl2.SDL_Event()
    try:
        # Play loop.
        while running:
            # Start of this visible frame.
            started = time.perf_counter()
            # Drain the SDL queue.
            while sdl2.SDL_PollEvent(event):
                # Quit from the window close button.
                if event.type == sdl2.SDL_QUIT:
                    # Leave the loop.
                    running = False
                # Key pressed.
                elif event.type == sdl2.SDL_KEYDOWN:
                    # F5 writes the snapshot.
                    if event.key.keysym.sym == sdl2.SDLK_F5:
                        # Save under the requested name.
                        _save(backend, pack, args.name)
                    # Esc quits.
                    elif event.key.keysym.sym == sdl2.SDLK_ESCAPE:
                        # Leave the loop.
                        running = False
                    # A game key.
                    elif event.key.keysym.sym in mapping and not event.key.repeat:
                        # Hold it.
                        held.add(mapping[event.key.keysym.sym])
                # Key released.
                elif event.type == sdl2.SDL_KEYUP and event.key.keysym.sym in mapping:
                    # Release it.
                    held.discard(mapping[event.key.keysym.sym])
            # Apply the held keys for this frame.
            backend.release_all()
            # Each held control.
            for player, name in held:
                # Press on that joystick.
                backend.press_player(player, name)
            # One game frame.
            backend.tick(1)
            # Show it.
            window.render_rgb(backend.get_rgb_frame())
            # Cap the play window near 60 frames per second.
            remaining = _FRAME_SECONDS - (time.perf_counter() - started)
            # Sleep only when the frame was faster than real time.
            if remaining > 0:
                # Seconds to milliseconds.
                time.sleep(remaining)
    finally:
        # Optional save on the way out.
        if args.save_on_exit:
            # Same file name as F5.
            _save(backend, pack, args.name)
        # Close the preview.
        window.close()
        # Stop MAME.
        backend.stop()


def main():
    """CLI entry."""
    # Parse and play.
    run(_parse_args())


if __name__ == "__main__":
    # Play and save.
    main()
