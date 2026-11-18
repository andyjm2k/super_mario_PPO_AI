"""Two-player MAME matches. One policy plays both sides of the screen."""

from __future__ import annotations

import copy
import os
import random
from typing import Callable, List, Optional

import numpy as np
from gymnasium import Env, Wrapper
from gymnasium.spaces import Box, Discrete

from observation_pipeline import preprocess_frame
from pdt.emulator import BOOT_STATE
from pdt.emulator.factory import create_emulator_backend, normalize_engine_name
from pdt.game_pack import GamePack, load_game_pack
from pdt.memory_bank import MemoryBank
from pdt.reward_engine import DeclarativeRewardEngine, RewardSpec


def mirror_health_spec(spec: RewardSpec) -> RewardSpec:
    """Return a reward spec with P1 and P2 health signals swapped."""
    # Copy so the player-1 document stays unchanged.
    data = copy.deepcopy(spec.data)
    # Each rule may name a health signal.
    for rule in data.get("rules", []):
        # Signal and the round-clock gate are the only name fields.
        for key in ("signal", "require_signal"):
            # Current name, if this rule has one.
            name = rule.get(key)
            # Player 1's bar becomes player 2's bar.
            if name == "P1_HEALTH":
                # Swapped.
                rule[key] = "P2_HEALTH"
            # Player 2's bar becomes player 1's bar.
            elif name == "P2_HEALTH":
                # Swapped.
                rule[key] = "P1_HEALTH"
    # A spec with no path of its own.
    return RewardSpec(data)


def orient_frame(frame: np.ndarray, side: int) -> np.ndarray:
    """Put the acting player's character on the left of the picture."""
    # Copy into a contiguous RGB array.
    image = np.ascontiguousarray(frame, dtype=np.uint8)
    # Player 2 faces the other way, so flip the shared screen.
    if int(side) == 2:
        # Left-right mirror.
        return np.ascontiguousarray(np.fliplr(image))
    # Player 1 already stands on the left at round start.
    return image


class MameVersusEnv(Env):
    """One arcade board. The stored trajectory alternates which fighter learns."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(
        self,
        game_id: str = "sfiii3n",
        games_root: str = "games",
        game_pack: Optional[GamePack] = None,
        state_paths: Optional[List[str]] = None,
        obs_width: Optional[int] = None,
        obs_height: Optional[int] = None,
        crop_top: Optional[int] = None,
        reward_spec: Optional[RewardSpec] = None,
        memory_bank: Optional[MemoryBank] = None,
        emulation_speed: float = 0,
        window: str = "null",
        backend=None,
        learner_side: Optional[int] = None,
        boot_actions: Optional[list] = None,
    ):
        # Gymnasium base.
        super().__init__()
        # Pack supplies the ROM, actions, and memory map.
        self.game_pack = game_pack or load_game_pack(game_id, games_root=games_root)
        # This env is the arcade board, not a link cable.
        engine = normalize_engine_name(self.game_pack.emulator_engine)
        # Refuse a GBA pack so the checkbox cannot boot the wrong core.
        if engine != "mame":
            # Tell the user which pack was selected.
            raise ValueError(
                f"P1 vs P2 training requires emulator 'mame', not '{engine}'."
            )
        # Start states. A saved .sta is preferred; boot is the fallback.
        self.state_paths = list(state_paths or self.game_pack.discover_states())
        # CNN width.
        self.obs_width = int(obs_width if obs_width is not None else self.game_pack.obs_width)
        # CNN height.
        self.obs_height = int(
            obs_height if obs_height is not None else self.game_pack.obs_height
        )
        # HUD crop. Street Fighter keeps the whole picture.
        self.crop_top = int(crop_top if crop_top is not None else self.game_pack.crop_top)
        # Discrete combos shared by both fighters.
        self.action_maps = list(self.game_pack.actions)
        # One action per step: the fighter whose trajectory is being stored.
        self.action_space = Discrete(len(self.action_maps))
        # Single grayscale frame. The LSTM carries time when the stack is 1.
        self.observation_space = Box(
            low=0,
            high=255,
            shape=(self.obs_height, self.obs_width, 1),
            dtype=np.uint8,
        )
        # Shared RAM map.
        self.memory_bank = memory_bank or self.game_pack.load_memory()
        # Player 1's reward document.
        loaded = reward_spec or self.game_pack.load_rewards()
        # Player 1 engine.
        self.reward_p1 = DeclarativeRewardEngine(loaded, self.memory_bank)
        # Player 2 sees the health bars swapped.
        self.reward_p2 = DeclarativeRewardEngine(mirror_health_spec(loaded), self.memory_bank)
        # Frames to hold each combo.
        self.wait_frames = self.game_pack.wait_frames
        # Live policy for the other fighter. None repeats the previous action.
        self._opponent_fn: Optional[Callable] = None
        # Path to a saved policy the subprocess workers can reload.
        self._opponent_path: Optional[str] = None
        # mtime of the file currently loaded, so we do not reload every step.
        self._opponent_mtime: Optional[float] = None
        # CPU copy of the other fighter's policy.
        self._opponent_model = None
        # Previous learner action, used when no policy is bound yet.
        self._lag_action = 0
        # Fixed side in tests. None rolls a side every episode.
        self._fixed_side = None if learner_side is None else int(learner_side)
        # Which fighter's return SB3 will store this episode.
        self.learner_side = self._fixed_side or 1
        # One MAME process, or a fake session in tests.
        self.emu = backend or self._open_backend(window, emulation_speed)
        # Coin sequence only for a cold boot. A snapshot skips it.
        coins = list(self.game_pack.meta.get("boot_actions", []) if boot_actions is None else boot_actions)
        # The backend stores the list for load_state_file.
        if hasattr(self.emu, "set_boot_actions"):
            # Hand the list through.
            self.emu.set_boot_actions(coins)
        # Unflipped picture for the training booth.
        self.last_frame = None
        # Latest info dict.
        self.last_info = {}
        # Worker id.
        self.agent_id = random.randint(1, 1_000_000)
        # Episode return of the learning fighter.
        self.total_reward = 0.0
        # Steps this episode.
        self.episode_length = 0
        # Which snapshot this match loaded.
        self.active_state = None

    def _open_backend(self, window: str, emulation_speed: float):
        """Boot one MAME process for this match."""
        # Factory selects MameBackend from the pack engine id.
        backend = create_emulator_backend(
            engine=self.game_pack.emulator_engine,
            rom_path=self.game_pack.rom_path,
            window=window,
            emulation_speed=emulation_speed,
        )
        # Watch the health and fight-flag addresses.
        if hasattr(backend, "bind_memory_bank"):
            # Register the pack map before the first tick.
            backend.bind_memory_bank(self.memory_bank)
        # Driver name, sfiii3n rather than the folder id.
        if hasattr(backend, "game_id") and self.game_pack.meta.get("mame_game"):
            # MAME's romset id.
            backend.game_id = str(self.game_pack.meta.get("mame_game"))
        # The constructed board.
        return backend

    def set_opponent(self, opponent_fn: Optional[Callable] = None) -> None:
        """Use a callable or a saved-model path for the other fighter."""
        # Subprocess workers receive a path because a live model will not pickle.
        if isinstance(opponent_fn, str):
            # Remember the zip. It is loaded on the next action.
            self._opponent_path = opponent_fn
            # Drop a direct callable so the file wins.
            self._opponent_fn = None
            # Force a reload if this is a new path.
            self._opponent_mtime = None
            # Done.
            return
        # In-process tests and the dummy vec env pass a function.
        self._opponent_fn = opponent_fn
        # A function replaces the file.
        self._opponent_path = None

    def _predict_saved(self, obs):
        """Load the latest opponent zip on CPU and return one action."""
        # Nothing saved yet.
        if not self._opponent_path or not os.path.isfile(self._opponent_path):
            # Caller falls back to lag-1.
            return None
        # Skip the reload when the file has not changed.
        mtime = os.path.getmtime(self._opponent_path)
        # First use, or the trainer wrote a newer zip.
        if self._opponent_model is None or mtime != self._opponent_mtime:
            # Import here so unit tests do not load torch until a real zip exists.
            from sb3_contrib import RecurrentPPO
            from stable_baselines3 import PPO
            try:
                # The default algorithm is recurrent.
                self._opponent_model = RecurrentPPO.load(self._opponent_path, device="cpu")
            except Exception:
                # Plain PPO checkpoints use the other loader.
                self._opponent_model = PPO.load(self._opponent_path, device="cpu")
            # Remember which file this object came from.
            self._opponent_mtime = mtime
        try:
            # CPU inference. The 5090 stays with the learner in the parent process.
            action, _state = self._opponent_model.predict(obs, deterministic=False)
        except Exception:
            # A bad checkpoint must not kill the match.
            return None
        # Numpy scalars and length-1 arrays both become an int.
        if hasattr(action, "reshape"):
            # First element.
            action = action.reshape(-1)[0]
        # Discrete index.
        return int(action) % len(self.action_maps)

    def _buttons(self, action: int) -> List[str]:
        """Button names for one discrete action."""
        # Clamp into the pack list.
        entry = self.action_maps[int(action) % len(self.action_maps)]
        # Names the pad table understands.
        return [str(name) for name in entry.get("buttons", [])]

    def _press(self, player: int, action: int) -> None:
        """Hold one fighter's combo. The caller cleared the pads first."""
        # Each named button.
        for name in self._buttons(action):
            # Player 1 or player 2 on the same board.
            self.emu.press_player(int(player), name)

    def _obs_for(self, side: int):
        """Grayscale view with that fighter on the left."""
        # Orient then crop.
        return preprocess_frame(
            orient_frame(self.emu.get_rgb_frame(), side),
            crop_top=self.crop_top,
            obs_width=self.obs_width,
            obs_height=self.obs_height,
        )

    def _other_action(self, obs, fallback: int) -> int:
        """Ask the live policy for the other fighter, else repeat the last action."""
        # A saved zip is how subprocess workers see the other fighter.
        if self._opponent_path:
            # CPU predict from the last checkpoint.
            predicted = self._predict_saved(obs)
            # Use it when the file loaded.
            if predicted is not None:
                # Discrete index.
                return int(predicted) % len(self.action_maps)
        # No policy bound yet.
        if self._opponent_fn is None:
            # Lag-1 keeps the match moving.
            return int(fallback) % len(self.action_maps)
        try:
            # Predict may return a numpy scalar.
            return int(self._opponent_fn(obs)) % len(self.action_maps)
        except Exception:
            # A bad predict must not kill the episode.
            return int(fallback) % len(self.action_maps)

    def step(self, action):
        """Act for both fighters and return the learning fighter's transition."""
        # The action SB3 chose belongs to the learning side.
        learner_action = int(action) % len(self.action_maps)
        # The other joystick.
        other = 1 if self.learner_side == 2 else 2
        # The other fighter sees the screen oriented for them.
        other_action = self._other_action(self._obs_for(other), self._lag_action)
        # Drop the previous combo.
        self.emu.release_all()
        # Learning fighter.
        self._press(self.learner_side, learner_action)
        # The other fighter.
        self._press(other, other_action)
        # Advance the board.
        self.emu.tick(self.wait_frames)
        # Count the step before the reward rules that use episode length.
        self.episode_length += 1
        # Player 1's shaped return.
        reward_1, term_1, trunc_1, info_1 = self.reward_p1.step(
            self.emu.memory, episode_length=self.episode_length
        )
        # Player 2's shaped return from the swapped health rules.
        reward_2, term_2, trunc_2, _info_2 = self.reward_p2.step(
            self.emu.memory, episode_length=self.episode_length
        )
        # Keep the learning fighter's numbers.
        reward = reward_1 if self.learner_side == 1 else reward_2
        # A knockout on either side ends the round.
        terminated = bool(term_1 or term_2)
        # Timeout from either engine.
        truncated = bool(trunc_1 or trunc_2)
        # This action is the lag if the policy is not bound next step.
        self._lag_action = learner_action
        # Running return.
        self.total_reward += float(reward)
        # Start from player 1's component dict and tag the match.
        info = dict(info_1)
        # Which side produced the stored reward.
        info["learner_side"] = int(self.learner_side)
        # The other fighter's action.
        info["opponent_action"] = int(other_action)
        # Both returns, for the booth and for debugging.
        info["reward_p1"] = float(reward_1)
        # Player 2.
        info["reward_p2"] = float(reward_2)
        # Worker id.
        info["agent_id"] = self.agent_id
        # Snapshot path.
        info["start_state"] = self.active_state
        # Pack id.
        info["game_id"] = self.game_pack.game_id
        # Always MAME here.
        info["emulator"] = "mame"
        # Mark the episode as a versus match.
        info["marl"] = True
        # Episode return of the learning fighter.
        info["episode_reward"] = self.total_reward
        # Episode length.
        info["episode_length"] = self.episode_length
        # Booth shows the real screen, not the flipped learner view.
        self.last_frame = np.array(self.emu.get_rgb_frame(), dtype=np.uint8, copy=True)
        # Cache for the training callback.
        self.last_info = info
        # Learning fighter's view.
        observation = self._obs_for(self.learner_side)
        # Gymnasium 5-tuple.
        return observation, float(reward), terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Load a start state and pick which fighter's return is stored."""
        # Gymnasium seeding.
        super().reset(seed=seed)
        # Tests pin the side. Training rolls it so both fighters learn.
        if self._fixed_side is None:
            # Either joystick.
            self.learner_side = random.choice((1, 2))
        else:
            # Pinned.
            self.learner_side = int(self._fixed_side)
        # Clear the episode counters.
        self.total_reward = 0.0
        # Steps.
        self.episode_length = 0
        # Do not reuse the previous match's buttons.
        self._lag_action = 0
        # Pick a snapshot, or the boot sentinel.
        selected = random.choice(self.state_paths) if self.state_paths else BOOT_STATE
        # Remember it.
        self.active_state = selected
        # Load the snapshot, or insert coins when the sentinel is selected.
        self.emu.load_state_file(selected)
        # Publish RAM from the loaded frame before the reward baselines are taken.
        self.emu.tick(1)
        # Seed both reward baselines from the loaded RAM.
        self.reward_p1.reset(self.emu.memory)
        # Player 2's baselines are independent.
        self.reward_p2.reset(self.emu.memory)
        # Unflipped picture.
        self.last_frame = np.array(self.emu.get_rgb_frame(), dtype=np.uint8, copy=True)
        # Info for the first observation.
        info = {
            "start_state": self.active_state,
            "game_id": self.game_pack.game_id,
            "emulator": "mame",
            "learner_side": int(self.learner_side),
            "marl": True,
        }
        # Cache it.
        self.last_info = info
        # Learning fighter's view.
        return self._obs_for(self.learner_side), info

    def close(self):
        """Stop the arcade board."""
        # The fake session and MAME both implement stop.
        self.emu.stop()


class OpponentHook(Wrapper):
    """Expose set_opponent on the wrapped env SB3's vec env can see."""

    def set_opponent(self, opponent_fn: Optional[Callable]) -> None:
        """Forward the live policy down to the versus env."""
        # Start at the wrapped env.
        current = self.env
        # Walk gym wrappers until the versus env.
        while current is not None and not hasattr(current, "set_opponent"):
            # Next inner env.
            current = getattr(current, "env", None)
        # The versus env records the callable.
        if current is not None and current is not self:
            # Bind it.
            current.set_opponent(opponent_fn)
