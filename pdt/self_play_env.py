"""Self-play environment: one policy, two GBAs, one local link cable."""

from __future__ import annotations

import random
from typing import Callable, List, Optional

import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box, Discrete

from observation_pipeline import preprocess_frame
from pdt.emulator.gba_link import GbaLink
from pdt.game_pack import GamePack, load_game_pack
from pdt.memory_bank import MemoryBank
from pdt.reward_engine import DeclarativeRewardEngine, RewardSpec


class SelfPlayEnv(Env):
    """mGBA link match. The learner is player 1; player 2 is the same policy."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(
        self,
        game_id: str = "street_fighter3",
        games_root: str = "games",
        game_pack: Optional[GamePack] = None,
        state_paths: Optional[List[str]] = None,
        obs_width: Optional[int] = None,
        obs_height: Optional[int] = None,
        crop_top: Optional[int] = None,
        reward_spec: Optional[RewardSpec] = None,
        memory_bank: Optional[MemoryBank] = None,
        emulation_speed: float = 0,
        opponent_fn: Optional[Callable] = None,
    ):
        # Gymnasium base.
        super().__init__()
        # Pack supplies ROM, actions, and whether link play is allowed.
        self.game_pack = game_pack or load_game_pack(game_id, games_root=games_root)
        # Refuse packs that did not opt into a link-versus mode.
        if not self.game_pack.link_play:
            # Mario and other single-player packs stay on the normal env.
            raise ValueError(
                f"Pack '{self.game_pack.game_id}' has no link_play support. "
                "Set \"link_play\": true in game.json for a link-cable game."
            )
        # Only mGBA exposes the lockstep cable used here.
        from pdt.emulator.factory import normalize_engine_name
        # Accept mgba and libmgba.
        if normalize_engine_name(self.game_pack.emulator_engine) != "mgba":
            # PyBoy and PyBoyAdvance cannot host this cable.
            raise ValueError(
                f"Self-play link requires emulator 'mgba', "
                f"not '{self.game_pack.emulator_engine}'."
            )
        # Start states, including the boot sentinel.
        self.state_paths = list(state_paths or self.game_pack.discover_states())
        # CNN width, pack default unless the UI overrode it.
        self.obs_width = int(obs_width if obs_width is not None else self.game_pack.obs_width)
        # CNN height.
        self.obs_height = int(
            obs_height if obs_height is not None else self.game_pack.obs_height
        )
        # HUD crop shared with single-player training.
        self.crop_top = int(crop_top if crop_top is not None else self.game_pack.crop_top)
        # Discrete actions from the pack (gameplay buttons only).
        self.action_maps = list(self.game_pack.actions)
        # Same action space as a single-player episode.
        self.action_space = Discrete(len(self.action_maps))
        # Player 1's pixels are the observation the policy learns from.
        self.observation_space = Box(
            low=0,
            high=255,
            shape=(self.obs_height, self.obs_width, 1),
            dtype=np.uint8,
        )
        # Memory map shared by both GBAs (each has its own RAM).
        self.memory_bank = memory_bank or self.game_pack.load_memory()
        # Reward rules read player 1's memory.
        loaded = reward_spec or self.game_pack.load_rewards()
        # Engine instance for the learner side.
        self.reward_engine = DeclarativeRewardEngine(loaded, self.memory_bank)
        # Frames held per action.
        self.wait_frames = self.game_pack.wait_frames
        # Optional opponent policy. None means "repeat my previous action".
        self._opponent_fn = opponent_fn
        # Last learner action, used when no opponent policy is attached.
        self._lag_action = 0
        # Two linked GBAs.
        self.link = GbaLink(
            rom_path=self.game_pack.rom_path,
            bios_path=self.game_pack.bios_path,
            skip_bios=self.game_pack.skip_bios,
            emulation_speed=float(emulation_speed),
        )
        # Player 1 (learner) and player 2 (opponent).
        self.p1 = self.link.players[0]
        # Slave GBA on the same cable.
        self.p2 = self.link.players[1]
        # Side-by-side RGB for the training UI.
        self.last_frame = None
        # Latest info dict for the UI caption.
        self.last_info = {}
        # Worker id shown in logs.
        self.agent_id = random.randint(1, 1_000_000)
        # Episode counters.
        self.total_reward = 0.0
        # Steps this episode.
        self.episode_length = 0
        # Which savestate both GBAs loaded.
        self.active_state = None

    def set_opponent(self, opponent_fn: Optional[Callable]) -> None:
        """Use ``opponent_fn(obs) -> action`` for player 2. None restores lag-1."""
        # The trainer passes a predict closure when it can.
        self._opponent_fn = opponent_fn

    def _apply(self, backend, action: int) -> None:
        """Hold the buttons for one discrete action on one GBA."""
        # Release the previous combo first.
        backend.release_all()
        # Clamp into the pack action list.
        buttons = list(self.action_maps[int(action) % len(self.action_maps)].get("buttons", []))
        # Press each named button.
        for name in buttons:
            # The backend maps the name onto the GBA keypad.
            backend.press_button(str(name))

    def _opponent_action(self, opp_obs) -> int:
        """Pick player 2's action from the live policy, else the lagged learner."""
        # A bound predict function is true self-play.
        if self._opponent_fn is not None:
            # The callback owns the policy; we only need an int action.
            try:
                # Predict may return a numpy scalar.
                return int(self._opponent_fn(opp_obs)) % len(self.action_maps)
            except Exception:
                # Fall through to the lagged action if predict fails.
                pass
        # Lag-1: player 2 repeats the learner's previous choice.
        return int(self._lag_action) % len(self.action_maps)

    def _combined_frame(self):
        """Place both live screens side by side for the training UI."""
        # Player 1 RGB.
        left = np.asarray(self.p1.get_rgb_frame(), dtype=np.uint8)
        # Player 2 RGB.
        right = np.asarray(self.p2.get_rgb_frame(), dtype=np.uint8)
        # Match heights before concatenating.
        if left.shape[0] != right.shape[0]:
            # Fall back to player 1 if the panels differ.
            return left
        # Wide frame: learner on the left, opponent on the right.
        return np.concatenate([left, right], axis=1)

    def _obs_from(self, backend):
        """Crop and grayscale one GBA's live frame."""
        # Same pipeline as single-player training.
        return preprocess_frame(
            backend.get_rgb_frame(),
            crop_top=self.crop_top,
            obs_width=self.obs_width,
            obs_height=self.obs_height,
        )

    def step(self, action):
        """Apply the learner action, let player 2 answer, then advance the cable."""
        # Remember this action so the lagged opponent can reuse it next step.
        learner = int(action) % len(self.action_maps)
        # Opponent sees its own pixels, not the learner's.
        opp_action = self._opponent_action(self._obs_from(self.p2))
        # Hold player 1's buttons.
        self._apply(self.p1, learner)
        # Hold player 2's buttons on the other GBA.
        self._apply(self.p2, opp_action)
        # Step both cores together so the cable transfer can finish.
        self.link.tick(self.wait_frames)
        # This action becomes the lag for the next opponent choice.
        self._lag_action = learner
        # Learner observation.
        observation = self._obs_from(self.p1)
        # Count the step.
        self.episode_length += 1
        # Rewards come from player 1's RAM (damage dealt, KO, and so on).
        reward, terminated, truncated, info = self.reward_engine.step(
            self.p1.memory,
            episode_length=self.episode_length,
        )
        # Running return.
        self.total_reward += float(reward)
        # UI and logs.
        info["agent_id"] = self.agent_id
        # Which savestate this match loaded.
        info["start_state"] = self.active_state
        # Pack id.
        info["game_id"] = self.game_pack.game_id
        # Always mGBA for this env.
        info["emulator"] = "mgba"
        # Mark the episode as link self-play.
        info["self_play"] = True
        # Player 2's action, for debugging.
        info["opponent_action"] = int(opp_action)
        # Episode return.
        info["episode_reward"] = self.total_reward
        # Episode length.
        info["episode_length"] = self.episode_length
        # Side-by-side live frames.
        self.last_frame = self._combined_frame()
        # Cache for the training callback.
        self.last_info = info
        # Gymnasium 5-tuple.
        return observation, float(reward), terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Load the same savestate into both GBAs and clear the reward trackers."""
        # Gymnasium seeding.
        super().reset(seed=seed)
        # Pick a linked pair when the setup tool has saved both sides.
        from pdt.game_pack import discover_link_pairs
        # Complete p1/p2 files in the pack states folder.
        pairs = discover_link_pairs(self.game_pack.states_dir)
        # A real link fight has a different savestate on each GBA.
        if pairs:
            # Choose one recorded match.
            p1_path, p2_path = random.choice(pairs)
            # Remember player 1's file for the UI caption.
            self.active_state = p1_path
            # Restore the master GBA.
            self.p1.load_state_file(p1_path)
            # Restore the slave GBA. Do not copy player 1's state onto it.
            self.p2.load_state_file(p2_path)
        else:
            # No pair yet: both GBAs boot or share one solo snapshot.
            selected = random.choice(self.state_paths)
            # Remember it for the UI caption.
            self.active_state = selected
            # Player 1.
            self.p1.load_state_file(selected)
            # Player 2 gets the same snapshot until a link pair exists.
            self.p2.load_state_file(selected)
        # One frame so both screens are valid.
        self.link.tick(1)
        # Clear episode counters.
        self.total_reward = 0.0
        # New episode.
        self.episode_length = 0
        # Opponent starts from a no-op until the learner acts.
        self._lag_action = 0
        # Seed reward baselines from player 1's RAM.
        self.reward_engine.reset(self.p1.memory)
        # Side-by-side picture for the first UI poll.
        self.last_frame = self._combined_frame()
        # Observation the policy sees first.
        observation = self._obs_from(self.p1)
        # Info for the reset.
        info = {
            "start_state": selected,
            "game_id": self.game_pack.game_id,
            "emulator": "mgba",
            "self_play": True,
        }
        # Cache it.
        self.last_info = info
        # Gymnasium reset pair.
        return observation, info

    def close(self):
        """Unplug the cable and drop both cores."""
        # GbaLink.stop releases both backends.
        self.link.stop()
