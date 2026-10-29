"""Train Mario agents with SB3 / sb3-contrib and vectorized PyBoy envs."""

import multiprocessing as mp

from stable_baselines3 import PPO
from sb3_contrib import QRDQN, RecurrentPPO

import callback as cb
from env_wrappers import build_vec_env
from gpu_tune import configure_torch_gpu
from model_architecture import build_policy_kwargs, default_policy_for_algorithm
from training_config import TrainingConfig


def create_model(config: TrainingConfig, env):
    """Instantiate RecurrentPPO (default), PPO, or QRDQN from config."""
    # Apply cuDNN/TF32 tweaks and resolve the concrete torch device.
    device = configure_torch_gpu(config.device)
    # Shared TensorBoard log directory for all algorithms.
    tb_log = config.log_dir
    # Build architecture kwargs from the configurable layer settings.
    policy_kwargs = build_policy_kwargs(config)
    # Resolve the policy class when the config still has a stale default.
    policy = config.policy or default_policy_for_algorithm(config.algorithm)
    # Announce the architecture so runs are easy to compare in logs.
    print(f"Policy={policy} kwargs={ {k: v for k, v in policy_kwargs.items() if k != 'activation_fn'} }")
    # RecurrentPPO keeps an LSTM so policies generalize across level layouts.
    if config.algorithm == "RecurrentPPO":
        # Build RecurrentPPO with the configured CNN+LSTM policy.
        model = RecurrentPPO(
            policy,
            env,
            learning_rate=config.learning_rate,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            gamma=config.gamma,
            ent_coef=config.ent_coef,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tb_log,
            verbose=1,
            device=device,
        )
    # Classic PPO remains available for ablation / faster iteration.
    elif config.algorithm == "PPO":
        # Build stock SB3 PPO with the configured CNN policy.
        model = PPO(
            policy if policy != "CnnLstmPolicy" else "CnnPolicy",
            env,
            learning_rate=config.learning_rate,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            gamma=config.gamma,
            ent_coef=config.ent_coef,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tb_log,
            verbose=1,
            device=device,
        )
    # Quantile Regression DQN for sample-efficient discrete control ablations.
    elif config.algorithm == "QRDQN":
        # Build QR-DQN with a configurable CNN + MLP Q-network.
        model = QRDQN(
            policy if policy != "CnnLstmPolicy" else "CnnPolicy",
            env,
            learning_rate=config.learning_rate,
            gamma=config.gamma,
            buffer_size=100_000,
            learning_starts=10_000,
            batch_size=config.batch_size,
            train_freq=4,
            target_update_interval=1000,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tb_log,
            verbose=1,
            device=device,
        )
    else:
        # Reject unknown algorithm names early.
        raise ValueError(f"Unsupported algorithm: {config.algorithm}")
    # Optionally resume weights from a previous checkpoint.
    if config.resume_path:
        # Map algorithm names onto their SB3 / contrib classes.
        loaders = {"RecurrentPPO": RecurrentPPO, "PPO": PPO, "QRDQN": QRDQN}
        # Load the checkpoint with the matching algorithm class.
        model = loaders[config.algorithm].load(
            config.resume_path, env=env, device=device
        )
    # Return the configured learner.
    return model


def train_model(config: TrainingConfig = None, frame_queue=None, stop_event=None):
    """Run a full training loop; stop_event enables cooperative UI halt."""
    # Fall back to defaults when no config object was supplied.
    config = config or TrainingConfig()
    # Make sure checkpoint / log directories exist.
    config.ensure_dirs()
    # Print the multi-level start-state curriculum.
    states = config.resolve_state_paths()
    # Announce how many start states will be sampled.
    print(f"Game={config.game_id} Training with {len(states)} start state(s): {states}")
    # Announce parallelization settings.
    print(
        f"Algorithm={config.algorithm} n_envs={config.n_envs} "
        f"vec={config.vec_env_type}"
    )
    # Build the vectorized PyBoy environments (skip then stack).
    env = build_vec_env(config)
    # Track whether learn finished via user stop for the return message.
    stopped_by_user = False
    # Always close workers even if learn() raises or is interrupted.
    try:
        # Construct the selected SB3 / sb3-contrib algorithm.
        model = create_model(config, env)
        # Attach checkpointing / UI frame streaming / stop callback.
        train_callback = cb.TrainAndLoggingCallback(
            check_freq=config.checkpoint_freq,
            save_path=config.checkpoint_dir,
            frame_queue=frame_queue,
            stop_event=stop_event,
        )
        # Optimize the policy for the requested number of timesteps.
        model.learn(total_timesteps=config.total_timesteps, callback=train_callback)
        # Remember if the callback halted because of a UI stop request.
        stopped_by_user = bool(train_callback.stopped_by_user)
        # Choose a filename that reflects how training ended.
        if stopped_by_user:
            # Interrupted runs already saved stopped_model inside the callback.
            final_path = f"{config.checkpoint_dir}/stopped_model"
        else:
            # Persist a final snapshot after a full successful run.
            final_path = f"{config.checkpoint_dir}/final_{config.algorithm}"
            # Save the finished model zip.
            model.save(final_path)
    finally:
        # Close all emulator workers to avoid zombie SubprocVecEnv processes.
        env.close()
    # Return the path to the final / stopped snapshot.
    return final_path


if __name__ == "__main__":
    # Required on Windows when spawning SubprocVecEnv workers.
    mp.freeze_support()
    # Launch training with the default multi-env configuration.
    train_model(TrainingConfig())
