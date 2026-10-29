"""Training hyper-parameters for Pyboy Deep Trainer."""

from dataclasses import dataclass, field
from typing import List, Optional
import os

from observation_pipeline import (
    DEFAULT_CROP_TOP,
    DEFAULT_OBS_HEIGHT,
    DEFAULT_OBS_WIDTH,
)
from pdt.game_pack import default_game_id, load_game_pack
from pdt.memory_bank import MemoryBank
from pdt.reward_engine import RewardSpec


@dataclass
class TrainingConfig:
    """Configuration for multi-env PyBoy training runs."""

    # Selected game pack id under games/.
    game_id: str = field(default_factory=lambda: default_game_id() or "mario_deluxe")
    # Root folder that contains one subdirectory per game.
    games_root: str = "games"
    # Absolute or relative path to the GBC ROM file (filled from pack when empty).
    rom_path: str = ""
    # Directory that stores checkpoint zip files.
    checkpoint_dir: str = "./train/"
    # Directory that stores TensorBoard / Monitor logs.
    log_dir: str = "./logs/"
    # Number of parallel PyBoy workers (vectorized environments).
    n_envs: int = 16
    # Total environment steps to train for.
    total_timesteps: int = 2_000_000
    # Algorithm name: RecurrentPPO (default), PPO, or QRDQN.
    algorithm: str = "RecurrentPPO"
    # Policy class name used by SB3 / sb3-contrib.
    policy: str = "CnnLstmPolicy"
    # Learning rate for the optimizer.
    learning_rate: float = 2.5e-4
    # Rollout length per environment before each update.
    n_steps: int = 1024
    # Minibatch size used during policy optimization.
    batch_size: int = 2048
    # Number of epochs over each rollout.
    n_epochs: int = 8
    # Discount factor emphasizing longer-horizon progress.
    gamma: float = 0.99
    # Entropy bonus to keep exploration alive across levels.
    ent_coef: float = 0.01
    # How often MaxAndSkipEnv repeats an action on single frames.
    frame_skip: int = 4
    # Frame stack depth; None means auto (2 RecurrentPPO / 4 otherwise).
    frame_stack: Optional[int] = None
    # CNN observation width after aspect-preserving resize.
    obs_width: int = DEFAULT_OBS_WIDTH
    # CNN observation height after HUD crop + resize.
    obs_height: int = DEFAULT_OBS_HEIGHT
    # Top rows cropped from the GBC screen to remove the HUD.
    crop_top: int = DEFAULT_CROP_TOP
    # How often to save checkpoints (in callback steps).
    checkpoint_freq: int = 4096
    # Whether to show live agent frames in the training UI.
    show_visuals: bool = True
    # Maximum number of agent panes to render in the UI grid.
    visual_agents: int = 4
    # Optional path to resume training from an existing model zip.
    resume_path: Optional[str] = None
    # Explicit start-state paths; empty means auto-discover from the pack.
    state_paths: List[str] = field(default_factory=list)
    # VecEnv backend: "subproc" for true parallelism, "dummy" for debug.
    vec_env_type: str = "subproc"
    # Device string passed to SB3 ("auto", "cuda", or "cpu").
    device: str = "auto"
    # Actor MLP hidden sizes as a comma-separated string (e.g. "512,512").
    pi_arch: str = "512,512"
    # Critic MLP hidden sizes as a comma-separated string.
    vf_arch: str = "512,512"
    # Q-network MLP sizes for QRDQN (falls back to pi_arch when empty).
    q_arch: str = "512,512"
    # NatureCNN output feature dimension before the MLP / LSTM.
    features_dim: int = 512
    # LSTM hidden size used by RecurrentPPO.
    lstm_hidden_size: int = 512
    # Number of stacked LSTM layers for RecurrentPPO.
    n_lstm_layers: int = 1
    # Whether actor and critic share one LSTM in RecurrentPPO.
    shared_lstm: bool = False
    # Whether the critic keeps its own LSTM in RecurrentPPO.
    enable_critic_lstm: bool = True
    # MLP activation name: relu, tanh, elu, or leaky_relu.
    activation: str = "relu"
    # Orthogonal weight initialization for the policy networks.
    ortho_init: bool = True
    # Optional override path for rewards.json (UI edits / alternate profiles).
    rewards_path: Optional[str] = None
    # Optional override path for memory.json (imported maps).
    memory_path: Optional[str] = None

    def resolve_pack(self):
        """Load the selected game pack metadata."""
        # Resolve pack from games_root / game_id.
        return load_game_pack(self.game_id, games_root=self.games_root)

    def apply_pack_defaults(self):
        """Fill ROM / obs fields from the game pack when still at placeholders."""
        # Load pack metadata.
        pack = self.resolve_pack()
        # Always sync rom_path from the pack so games never co-mingle.
        self.rom_path = pack.rom_path
        # Obs settings: keep explicit UI values, but seed from pack on first use
        # when callers construct a bare config — pack values win if still default
        # and pack differs. Prefer pack whenever config matches module defaults
        # AND pack has them. Simpler: always take pack obs unless caller set
        # state_paths-only. For UI we set obs from pack on game change.
        return pack

    def resolve_state_paths(self):
        """Return the list of .state files used for multi-level starts."""
        # Prefer explicitly configured states when the user supplied them.
        if self.state_paths:
            # Keep only paths that currently exist on disk.
            return [path for path in self.state_paths if os.path.isfile(path)]
        # Discover states only inside the selected game pack.
        pack = self.resolve_pack()
        # Return pack-local curriculum states.
        return pack.discover_states()

    def load_memory_bank(self) -> MemoryBank:
        """Load memory addresses from override path or the game pack."""
        # Explicit imported / alternate memory map.
        if self.memory_path and os.path.isfile(self.memory_path):
            # Load the override file.
            return MemoryBank.from_file(self.memory_path)
        # Default: pack memory.json.
        return self.resolve_pack().load_memory()

    def load_reward_spec(self) -> RewardSpec:
        """Load declarative rewards from override path or the game pack."""
        # Explicit UI-saved / alternate rewards file.
        if self.rewards_path and os.path.isfile(self.rewards_path):
            # Load the override file.
            return RewardSpec.from_file(self.rewards_path)
        # Default: pack rewards.json.
        return self.resolve_pack().load_rewards()

    def ensure_dirs(self):
        """Create checkpoint and log directories when they are missing."""
        # Create the checkpoint directory for model snapshots.
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        # Create the log directory for Monitor / TensorBoard output.
        os.makedirs(self.log_dir, exist_ok=True)
        # Sync ROM path from the selected pack.
        self.apply_pack_defaults()
