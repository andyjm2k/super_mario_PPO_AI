"""Generic Gymnasium environment for any Pyboy Deep Trainer game pack."""

from __future__ import annotations

import os
import random
from typing import List, Optional

import cv2
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box, Discrete

from observation_pipeline import preprocess_frame
from pdt.emulator import BOOT_STATE
from pdt.emulator.factory import create_emulator_backend, normalize_engine_name
from pdt.game_pack import GamePack, load_game_pack
from pdt.memory_bank import MemoryBank
from pdt.reward_engine import DeclarativeRewardEngine, RewardSpec


class PyBoyEnv(Env):
    """Multi-start env driven by declarative packs + pluggable emulators."""

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 60}

    def __init__(
        self,
        game_id: str = "mario_deluxe",
        games_root: str = "games",
        game_pack: Optional[GamePack] = None,
        state_paths: Optional[List[str]] = None,
        window: str = "null",
        obs_width: Optional[int] = None,
        obs_height: Optional[int] = None,
        crop_top: Optional[int] = None,
        render_mode=None,
        reward_spec: Optional[RewardSpec] = None,
        memory_bank: Optional[MemoryBank] = None,
        emulation_speed: int = 0,
    ):
        # Initialize Gymnasium base class.
        super().__init__()
        # Resolve the game pack from id unless one was injected (tests).
        self.game_pack = game_pack or load_game_pack(game_id, games_root=games_root)
        # Absolute ROM path for the selected backend.
        self.rom_path = self.game_pack.rom_path
        # Canonical emulator engine id.
        self.engine = normalize_engine_name(self.game_pack.emulator_engine)
        # Curriculum of start states (may include BOOT_STATE).
        self.state_paths = self._resolve_states(state_paths)
        # Observation geometry prefers explicit overrides, else pack defaults.
        self.obs_width = int(obs_width if obs_width is not None else self.game_pack.obs_width)
        # Observation height.
        self.obs_height = int(
            obs_height if obs_height is not None else self.game_pack.obs_height
        )
        # HUD crop in pixels from the top of the screen.
        self.crop_top = int(crop_top if crop_top is not None else self.game_pack.crop_top)
        # Gymnasium render mode.
        self.render_mode = render_mode
        # Emulation speed: 0=unlimited train, 1=real-time watch.
        self.emulation_speed = float(emulation_speed)
        # Button maps for each discrete action.
        self.action_maps = list(self.game_pack.actions)
        # Discrete action space sized from the pack.
        self.action_space = Discrete(len(self.action_maps))
        # Single grayscale frame; wrappers stack later.
        self.observation_space = Box(
            low=0,
            high=255,
            shape=(self.obs_height, self.obs_width, 1),
            dtype=np.uint8,
        )
        # Load declarative memory addresses for this game.
        self.memory_bank = memory_bank or self.game_pack.load_memory()
        # Load declarative rewards (or accept a UI-edited override).
        loaded_rewards = reward_spec or self.game_pack.load_rewards()
        # Build the reward engine that interprets those rules.
        self.reward_engine = DeclarativeRewardEngine(loaded_rewards, self.memory_bank)
        # Window string remembered for recycle.
        self._window = window
        # Boot the selected emulator backend.
        self.emu = self._create_backend()
        # Backwards-compatible alias used by older tests / callbacks.
        self.pyboy = self.emu
        # Latest RGB frame for the training UI.
        self.last_frame = None
        # Latest info dict for callbacks / UI.
        self.last_info = {}
        # Unique worker id.
        self.agent_id = random.randint(1, 1_000_000)
        # Emulator recycle counter.
        self.pyboy_counter = 0
        # Frames held per action from the pack.
        self.wait_frames = self.game_pack.wait_frames
        # Initialize episode trackers.
        self._reset_episode_trackers()
        # Announce the worker.
        print(
            f"STARTED AGENT: {self.agent_id} game={self.game_pack.game_id} "
            f"engine={self.engine} states={len(self.state_paths)}"
        )

    def _create_backend(self):
        """Construct a fresh emulator backend for this pack."""
        # Delegate to the factory (PyBoy or PyBoyAdvance).
        return create_emulator_backend(
            engine=self.engine,
            rom_path=self.rom_path,
            window=self._window,
            emulation_speed=self.emulation_speed,
            bios_path=self.game_pack.bios_path,
            skip_bios=self.game_pack.skip_bios,
        )

    def _resolve_states(self, state_paths):
        """Resolve start states from an explicit list or the pack folder."""
        # Prefer caller-provided existing paths (plus boot sentinel).
        if state_paths:
            # Keep boot sentinel and files that exist.
            return [
                path
                for path in state_paths
                if path == BOOT_STATE or os.path.isfile(path)
            ]
        # Discover states inside the game pack.
        discovered = self.game_pack.discover_states()
        # Fail fast when the pack has no snapshots and no boot fallback.
        if not discovered:
            # Point authors at the expected folder / flags.
            raise FileNotFoundError(
                f"No start states found under {self.game_pack.states_dir}. "
                "Add *.state / *.pba_state files, or set "
                '"allow_boot_start": true in game.json.'
            )
        # Return the curriculum list.
        return discovered

    def _reset_episode_trackers(self):
        """Clear per-episode counters."""
        # Cumulative reward.
        self.total_reward = 0.0
        # Steps taken.
        self.episode_length = 0
        # Active start-state path.
        self.active_state = None
        # Clear last info.
        self.last_info = {}

    def step(self, action):
        """Advance one low-level action and return a Gymnasium transition."""
        # Press the buttons for this discrete action.
        self.execute_action(int(action))
        # Build the single-frame observation.
        observation = self.get_observation()
        # Count the agent step.
        self.episode_length += 1
        # Shape reward from declarative rules + live memory.
        reward, terminated, truncated, info = self.reward_engine.step(
            self.emu.memory,
            episode_length=self.episode_length,
        )
        # Accumulate return.
        self.total_reward += reward
        # Attach worker metadata.
        info["agent_id"] = self.agent_id
        # Attach start state.
        info["start_state"] = self.active_state
        # Attach game id.
        info["game_id"] = self.game_pack.game_id
        # Attach emulator engine id.
        info["emulator"] = self.engine
        # Attach episode return.
        info["episode_reward"] = self.total_reward
        # Attach episode length.
        info["episode_length"] = self.episode_length
        # Optional score signal when the pack defines SCORE.
        if "SCORE" in self.memory_bank.signals:
            # Read assembled score for logging only.
            info["score"] = int(self.memory_bank.read(self.emu.memory, "SCORE"))
        # Cache info for UI polling.
        self.last_info = info
        # Return the 5-tuple.
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Reload a random start state and seed the reward engine."""
        # Honor Gymnasium seeding.
        super().reset(seed=seed)
        # Periodically recycle the emulator to limit long-run leaks.
        if self.pyboy_counter >= 5000:
            # Stop the old instance.
            self.emu.stop()
            # Boot a fresh backend.
            self.emu = self._create_backend()
            # Keep the legacy alias in sync.
            self.pyboy = self.emu
            # Reset recycle counter.
            self.pyboy_counter = 0
        # Clear episode counters.
        self._reset_episode_trackers()
        # Load a random curriculum state.
        self.load_random_state()
        # Seed reward trackers from the post-load memory snapshot.
        self.reward_engine.reset(self.emu.memory)
        # Count this reset toward recycle budget.
        self.pyboy_counter += 1
        # Return observation + info.
        return self.get_observation(), {
            "start_state": self.active_state,
            "game_id": self.game_pack.game_id,
            "emulator": self.engine,
        }

    def render(self):
        """Return / display the latest RGB frame."""
        # Grab a fresh frame.
        frame = self.get_rgb_frame()
        # Optional OpenCV human window.
        if self.render_mode == "human":
            # Show BGR for OpenCV.
            cv2.imshow(
                f"PDT Agent {self.agent_id}",
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
            )
            # Pump events without blocking.
            cv2.waitKey(1)
        # Always return RGB.
        return frame

    def close(self):
        """Shut down OpenCV windows and the emulator."""
        # Destroy OpenCV windows.
        cv2.destroyAllWindows()
        # Stop the backend.
        self.emu.stop()

    def get_rgb_frame(self):
        """Read the emulator screen as an RGB uint8 array."""
        # Delegate to the backend.
        raw = self.emu.get_rgb_frame()
        # Cache for the UI.
        self.last_frame = raw
        # Return RGB.
        return raw

    def get_observation(self):
        """Return one cropped/resized grayscale frame."""
        # Start from RGB.
        rgb = self.get_rgb_frame()
        # Apply pack / config crop + resize.
        return preprocess_frame(
            rgb,
            crop_top=self.crop_top,
            obs_width=self.obs_width,
            obs_height=self.obs_height,
        )

    def execute_action(self, action: int):
        """Map a discrete action id onto emulator button presses."""
        # Clear previous holds.
        self.emu.release_all()
        # Clamp to a valid action index.
        index = int(action) % len(self.action_maps)
        # Fetch the button list for this action.
        buttons = list(self.action_maps[index].get("buttons", []))
        # Press each configured button.
        for button in buttons:
            # Hold the named button via the backend.
            self.emu.press_button(str(button))
        # Advance while holding.
        self.emu.tick(count=self.wait_frames)

    def load_random_state(self):
        """Load a random saved state from the pack curriculum.

        GBA backends warm-boot inside ``load_state_file`` then inject the
        ``.pba_state`` (RAM + frozen rgb_frame). Avoid an extra post-load
        tick that can blank the PPU or race incomplete restores.
        """
        # Sample one start state.
        selected = random.choice(self.state_paths)
        # Remember the path.
        self.active_state = selected
        # Restore the snapshot (or boot); GBA path boots then applies state.
        self.emu.load_state_file(selected)
        # GB/GBC needs a tick so PyBoy's screen buffer is valid after load.
        if self.engine != "pyboy_advance":
            # Advance one frame after a native PyBoy savestate.
            self.emu.tick(1)

    def release_all_keys(self):
        """Release movement / action buttons before a new combo."""
        # Delegate to the backend.
        self.emu.release_all()


# Backwards-compatible alias used by older imports / tests.
GbaGame = PyBoyEnv
