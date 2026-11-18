"""Two-player link-cable window used to record a self-play savestate pair.

Player 1 (left screen):
  Arrows move, A/S = A/B, Q/W = L/R, Enter = Start, Backspace = Select

Player 2 (right screen):
  I/J/K/L move, U/O = A/B, Y/P = L/R, N = Start, M = Select

F5 saves states/<name>_p1.mgba_state and states/<name>_p2.mgba_state.
Esc quits.
"""

from __future__ import annotations

import ctypes
import os

import numpy as np
import sdl2
import sdl2.ext

from pdt.emulator.gba_link import GbaLink
from pdt.emulator.mgba_backend import save_mgba_state

# Native GBA width for one screen.
_GBA_W = 240
# Native GBA height.
_GBA_H = 160
# Side-by-side width.
_PAIR_W = _GBA_W * 2
# Window scale. Two panels at 3x is 1440x480.
_WINDOW_SCALE = 3
# Character select ignores a tap shorter than this many frames.
_TAP_HOLD_FRAMES = 18


class InputLatch:
    """Remember buttons long enough for the link select screen to see them."""

    def __init__(self, hold_frames: int = _TAP_HOLD_FRAMES):
        # How many frames a tap stays down after the physical key comes up.
        self.hold_frames = int(hold_frames)
        # Buttons the keyboard is currently holding.
        self.down = set()
        # Extra frames to keep a tap visible after key-up.
        self.force = {}

    def note_down(self, player: int, name: str) -> None:
        """Record a press and keep it down across several frames."""
        # Physical key is down.
        self.down.add((int(player), name))
        # A fresh tap restarts the hold so a second press still counts.
        self.force[(int(player), name)] = self.hold_frames

    def note_up(self, player: int, name: str) -> None:
        """Record a release. The button stays down until the hold expires."""
        # The keyboard is no longer holding it.
        self.down.discard((int(player), name))

    def held(self, player: int) -> set:
        """Button names that should be down on this GBA for the next frame."""
        # Start from keys still physically held.
        names = {name for owner, name in self.down if owner == int(player)}
        # Taps that have not yet been visible for long enough.
        for (owner, name), left in self.force.items():
            # This side still owes the game a few frames of this button.
            if owner == int(player) and left > 0:
                # Keep it down.
                names.add(name)
        # The set the core should receive.
        return names

    def age(self) -> None:
        """Count one frame against every tap hold."""
        # Drop holds that have been visible long enough.
        for key in list(self.force):
            # One frame consumed.
            self.force[key] -= 1
            # The tap has lasted long enough.
            if self.force[key] <= 0:
                # Forget it. A key that is still down stays via self.down.
                del self.force[key]


def build_p2_key_maps():
    """Return (keydown, keyup) maps for the right-hand GBA."""
    # Right-hand cluster so it does not collide with player 1.
    key_down = {
        sdl2.SDLK_u: "a",
        sdl2.SDLK_o: "b",
        sdl2.SDLK_m: "select",
        sdl2.SDLK_n: "start",
        sdl2.SDLK_l: "right",
        sdl2.SDLK_j: "left",
        sdl2.SDLK_i: "up",
        sdl2.SDLK_k: "down",
        sdl2.SDLK_p: "r",
        sdl2.SDLK_y: "l",
    }
    # Releases use the same keys.
    key_up = dict(key_down)
    # Both maps.
    return key_down, key_up


def _save_pair(link, pack, name: str):
    """Write both GBAs' full savestates."""
    # Ensure the states folder exists.
    os.makedirs(pack.states_dir, exist_ok=True)
    # Player 1 path.
    p1_path = os.path.join(pack.states_dir, f"{name}_p1.mgba_state")
    # Player 2 path.
    p2_path = os.path.join(pack.states_dir, f"{name}_p2.mgba_state")
    # Master GBA (left screen).
    save_mgba_state(link.players[0], p1_path)
    # Slave GBA (right screen).
    save_mgba_state(link.players[1], p2_path)
    # Tell the user both files landed.
    print(f"Saved link pair -> {p1_path}")
    # Second line for player 2.
    print(f"                 {p2_path}")
    # Both paths.
    return p1_path, p2_path


class _PairWindow:
    """SDL window that shows both linked GBAs side by side."""

    def __init__(self):
        # Title mentions which side is which.
        self._title = b"mGBA link  |  left P1  |  right P2"
        # Filled in __enter__.
        self._window = None
        # Renderer.
        self._renderer = None
        # One texture for both panels.
        self._texture = None

    def __enter__(self):
        # Video only. The cores stay headless.
        if sdl2.SDL_InitSubSystem(sdl2.SDL_INIT_VIDEO) < 0:
            # Surface the SDL error.
            raise RuntimeError(f"SDL_InitSubSystem failed: {sdl2.SDL_GetError().decode()}")
        # Wide window for two panels.
        self._window = sdl2.SDL_CreateWindow(
            self._title,
            sdl2.SDL_WINDOWPOS_CENTERED,
            sdl2.SDL_WINDOWPOS_CENTERED,
            _PAIR_W * _WINDOW_SCALE,
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
        # Logical size is two GBA screens.
        sdl2.SDL_RenderSetLogicalSize(self._renderer, _PAIR_W, _GBA_H)
        # RGB24 texture covering both panels.
        self._texture = sdl2.SDL_CreateTexture(
            self._renderer,
            sdl2.SDL_PIXELFORMAT_RGB24,
            sdl2.SDL_TEXTUREACCESS_STREAMING,
            _PAIR_W,
            _GBA_H,
        )
        # Fail if the texture could not be created.
        if not self._texture:
            # Bubble the SDL error.
            raise RuntimeError(f"SDL_CreateTexture failed: {sdl2.SDL_GetError().decode()}")
        # Show the window.
        sdl2.SDL_ShowWindow(self._window)
        # Return self for ``with``.
        return self

    def set_title(self, title: str) -> None:
        """Update the title bar so the current control mode is visible."""
        # SDL wants bytes.
        sdl2.SDL_SetWindowTitle(self._window, title.encode("utf-8"))

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Destroy the texture when it exists.
        if self._texture:
            # Free it.
            sdl2.SDL_DestroyTexture(self._texture)
        # Destroy the renderer when it exists.
        if self._renderer:
            # Free it.
            sdl2.SDL_DestroyRenderer(self._renderer)
        # Destroy the window when it exists.
        if self._window:
            # Free it.
            sdl2.SDL_DestroyWindow(self._window)
        # Shut down the video subsystem this tool started.
        sdl2.SDL_QuitSubSystem(sdl2.SDL_INIT_VIDEO)

    def render_pair(self, left: np.ndarray, right: np.ndarray) -> None:
        """Blit player 1 on the left and player 2 on the right."""
        # Contiguous RGB panels.
        left_frame = np.ascontiguousarray(left, dtype=np.uint8)
        # Player 2.
        right_frame = np.ascontiguousarray(right, dtype=np.uint8)
        # Both must be native GBA frames.
        if left_frame.shape != (_GBA_H, _GBA_W, 3) or right_frame.shape != (_GBA_H, _GBA_W, 3):
            # Refuse a mismatched buffer.
            raise ValueError("Both link screens must be 160x240 RGB")
        # Side-by-side image.
        both = np.concatenate([left_frame, right_frame], axis=1)
        # Pointer to the first byte.
        ptr = both.ctypes.data_as(ctypes.c_void_p)
        # Pitch is the combined width times 3 bytes.
        sdl2.SDL_UpdateTexture(self._texture, None, ptr, _PAIR_W * 3)
        # Clear.
        sdl2.SDL_RenderClear(self._renderer)
        # Copy.
        sdl2.SDL_RenderCopy(self._renderer, self._texture, None, None)
        # Present.
        sdl2.SDL_RenderPresent(self._renderer)


def _handle_key(link, key, key_map, player_index: int, press: bool) -> None:
    """Press or release a button on the GBA that owns this key."""
    # Player 1 map is checked by the caller via which map matched.
    mapped = key_map.get(key)
    # Not a gameplay key.
    if mapped is None:
        # Ignore it.
        return
    # Which GBA this map belongs to is passed in. A plain dict cannot store it.
    player = link.players[player_index]
    # Press or release.
    if press:
        # Hold the button.
        player.press_button(mapped)
    else:
        # Release the button.
        player.release_button(mapped)


def run_link_setup(args) -> None:
    """Play two linked GBAs and save both savestates on F5."""
    # Local import avoids a cycle when the solo tool imports this module.
    from pdt.game_pack import load_game_pack
    from pdt.emulator.factory import normalize_engine_name
    from tools.save_mgba_state import build_key_maps

    # Load the pack named on the command line.
    pack = load_game_pack(args.game, games_root=args.games_root)
    # This mode is mGBA only.
    if normalize_engine_name(pack.emulator_engine) != "mgba":
        # Other engines have no lockstep cable.
        raise SystemExit(
            f"Pack '{pack.game_id}' uses '{pack.emulator_engine}'. "
            "Link setup requires emulator mgba."
        )
    # Player 1 keys. The player index is passed into the handler, not stored on the dict.
    p1_down, p1_up = build_key_maps()
    # Player 2 keys.
    p2_down, p2_up = build_p2_key_maps()
    # Remember whether F5 ran.
    saved_once = False
    # Print the controls once.
    print(
        f"Link setup for {pack.name}\n"
        "  LEFT  P1: Arrows move | A/S = A/B | Q/W = L/R | Enter start | Backspace select\n"
        "  RIGHT P2: IJKL move   | U/O = A/B | Y/P = L/R | N start      | M select\n"
        "  F6 mirrors P1 onto both | F5 saves both sides | Esc quits\n"
        "Get both sides into the link fight, then press F5."
    )
    # Two cores plus the cable. This can take a moment on first boot.
    link = GbaLink(
        rom_path=pack.rom_path,
        bios_path=pack.bios_path,
        skip_bios=pack.skip_bios,
        emulation_speed=float(args.speed),
    )
    try:
        # One window, both screens.
        with _PairWindow() as window:
            # Main loop.
            running = True
            # Each keyboard cluster drives only its own GBA.
            mirror = False
            # Taps are held in software so the select cursor can see them.
            latch = InputLatch()
            # What is currently pushed into each core. Updated only between frames.
            applied = [set(), set()]
            # Title tells the user which mode is active.
            window.set_title("mGBA link | left P1, right P2 | F6 mirror | F5 save")

            def _note(key, key_map, index: int, press: bool) -> None:
                """Record a key on one side, and on both sides while mirrored."""
                # Button name for this physical key.
                name = key_map[key]
                # Which GBAs should see it.
                targets = [index]
                # Optional mirror onto player 2.
                if mirror and index == 0:
                    # The other screen.
                    targets.append(1)
                # Update the latch. The cores are touched later, between frames.
                for target in targets:
                    # Press and release go through the hold timer.
                    if press:
                        # Start or restart the tap.
                        latch.note_down(target, name)
                    else:
                        # The physical key is up. The hold may keep it down.
                        latch.note_up(target, name)

            def _push_held() -> None:
                """Write the latch into the cores while they are not mid-frame."""
                # Each GBA has its own held set.
                for index, player in enumerate(link.players):
                    # Buttons that should be down for the coming frame.
                    want = latch.held(index)
                    # Newly pressed since last push.
                    for name in want - applied[index]:
                        # Hold it on the core.
                        player.press_button(name)
                    # Newly released since last push.
                    for name in applied[index] - want:
                        # Let it go on the core.
                        player.release_button(name)
                    # Remember what the core has now.
                    applied[index] = set(want)

            def _pump() -> None:
                """Read SDL keys. Safe to call while a frame is still running."""
                # The loop flag, mirror mode, and save flag live in the enclosing scope.
                nonlocal running, mirror, saved_once
                # Every queued key, including ones that arrived mid-frame.
                for event in sdl2.ext.get_events():
                    # Window close.
                    if event.type == sdl2.SDL_QUIT:
                        # Stop.
                        running = False
                    # Key down.
                    elif event.type == sdl2.SDL_KEYDOWN:
                        # Ignore key repeat so a held key is one press.
                        if event.key.repeat:
                            # Skip it.
                            continue
                        # Raw keycode.
                        key = event.key.keysym.sym
                        # F6 toggles whether P1 keys also drive P2.
                        if key == sdl2.SDLK_F6:
                            # Flip the mode.
                            mirror = not mirror
                            # Show the mode on the title bar.
                            window.set_title(
                                "mGBA link | BOTH follow arrow keys | F6 split | F5 save"
                                if mirror
                                else "mGBA link | left P1, right P2 | F6 mirror | F5 save"
                            )
                        # F5 writes the pair.
                        elif key == sdl2.SDLK_F5:
                            # Save both GBAs.
                            _save_pair(link, pack, args.name)
                            # Remember the save.
                            saved_once = True
                        # Esc quits.
                        elif key == sdl2.SDLK_ESCAPE:
                            # Stop.
                            running = False
                        elif key in p1_down:
                            # Player 1 press, mirrored onto player 2 when mirror is on.
                            _note(key, p1_down, 0, True)
                        elif key in p2_down:
                            # Player 2 press.
                            _note(key, p2_down, 1, True)
                    # Key up.
                    elif event.type == sdl2.SDL_KEYUP:
                        # Raw keycode.
                        key = event.key.keysym.sym
                        # Esc on key-up as well.
                        if key == sdl2.SDLK_ESCAPE:
                            # Stop.
                            running = False
                        elif key in p1_up:
                            # Player 1 release, mirrored when mirror is on.
                            _note(key, p1_up, 0, False)
                        elif key in p2_up:
                            # Player 2 release.
                            _note(key, p2_up, 1, False)

            # Until Esc or the window closes.
            while running:
                # Push taps into the cores before they sample the keypad.
                _push_held()
                # Advance one frame and keep reading keys while the cores run.
                link.tick(1, poll=_pump)
                # A tap stays down for several frames after the key comes up.
                latch.age()
                # Leave after Esc / close, even if it arrived mid-frame.
                if not running:
                    # Stop.
                    break
                # Draw both live frames.
                window.render_pair(
                    link.players[0].get_rgb_frame(),
                    link.players[1].get_rgb_frame(),
                )
            # Optional save on exit.
            if args.save_on_exit:
                # Dump both sides.
                _save_pair(link, pack, args.name)
            elif not saved_once:
                # Reminder when F5 was never pressed.
                print(
                    "No F5 save. Get both GBAs into the link fight, then press F5."
                )
    finally:
        # Drop both cores.
        link.stop()
