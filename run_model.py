"""Evaluate a trained Mario policy with optional RecurrentPPO LSTM state."""

import argparse
import glob
import os

import numpy as np
from stable_baselines3 import PPO
from sb3_contrib import QRDQN, RecurrentPPO

from env_wrappers import build_eval_env
from observation_pipeline import resolve_frame_stack
from training_config import TrainingConfig


def list_checkpoints(checkpoint_dir="./train/"):
    """Return sorted .zip checkpoint paths under the train directory."""
    # Match zip files written by SB3 model.save().
    pattern = os.path.join(checkpoint_dir, "*.zip")
    # Also match bare save paths that SB3 may write without an extension.
    bare = glob.glob(os.path.join(checkpoint_dir, "best_*"))
    # Collect zip paths first.
    paths = glob.glob(pattern)
    # Include bare checkpoints that are actually zip containers.
    for path in bare:
        # Skip directories.
        if os.path.isdir(path):
            # Ignore non-files.
            continue
        # Skip duplicates already covered by *.zip.
        if path.endswith(".zip"):
            # Already included.
            continue
        # Keep bare checkpoint files.
        paths.append(path)
    # Sort newest first for the runner dropdown.
    paths.sort(key=lambda item: os.path.getmtime(item), reverse=True)
    # Return the discovered checkpoint list.
    return paths


def load_model(algorithm, model_path, env, device="auto"):
    """Load the correct algorithm class for a saved model zip."""
    # RecurrentPPO checkpoints need the contrib loader.
    if algorithm == "RecurrentPPO":
        # Load the recurrent policy and attach the eval env.
        return RecurrentPPO.load(model_path, env=env, device=device)
    # Stock PPO checkpoints use the core SB3 loader.
    if algorithm == "PPO":
        # Load the feed-forward PPO policy.
        return PPO.load(model_path, env=env, device=device)
    # QR-DQN checkpoints use the contrib loader.
    if algorithm == "QRDQN":
        # Load the quantile DQN policy.
        return QRDQN.load(model_path, env=env, device=device)
    # Reject unknown algorithm names.
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def run_episodes(
    model_path,
    algorithm="RecurrentPPO",
    num_episodes=10,
    render=True,
    deterministic=True,
    device="auto",
    frame_stack=None,
    stop_event=None,
    result_callback=None,
    game_id=None,
    games_root="games",
):
    """Roll out a trained agent; return a list of per-episode summaries."""
    # Resolve start states / obs settings the same way training does.
    config = TrainingConfig(
        game_id=game_id or TrainingConfig().game_id,
        games_root=games_root,
        algorithm=algorithm,
        frame_stack=resolve_frame_stack(algorithm, frame_stack),
        device=device,
    )
    # Sync ROM / obs from the selected game pack.
    config.ensure_dirs()
    # Fill obs from pack explicitly for eval.
    pack = config.resolve_pack()
    # Apply pack observation geometry.
    config.obs_width = pack.obs_width
    # Height.
    config.obs_height = pack.obs_height
    # HUD crop.
    config.crop_top = pack.crop_top
    # ROM path.
    config.rom_path = pack.rom_path
    # Build DummyVecEnv + MaxAndSkip + VecFrameStack matching training.
    env = build_eval_env(config, render=render)
    # Load the selected algorithm checkpoint.
    model = load_model(algorithm, model_path, env, device=device)
    # Collect per-episode results for the runner UI / CLI.
    results = []
    # Evaluate for the requested number of episodes.
    try:
        # Loop over evaluation episodes.
        for episode in range(num_episodes):
            # Honor cooperative stop from the runner UI.
            if stop_event is not None and stop_event.is_set():
                # Exit early when the user pressed Stop.
                break
            # Reset the vec env and obtain the first stacked observation.
            obs = env.reset()
            # Clear LSTM state at the start of every episode.
            lstm_states = None
            # Mark the first step as an episode boundary.
            episode_starts = np.ones((env.num_envs,), dtype=bool)
            # Episode bookkeeping.
            done = False
            # Accumulate the undiscounted return.
            total_reward = 0.0
            # Track best progress reached during the episode.
            best_progress = 0.0
            # Last info dict from the worker for the printed summary.
            last_info = {}
            # Step until termination or truncation.
            while not done:
                # Honor stop mid-episode as well.
                if stop_event is not None and stop_event.is_set():
                    # Break out of the step loop.
                    break
                # Recurrent policies need state + episode_start kwargs.
                if algorithm == "RecurrentPPO":
                    # Predict while updating LSTM hidden/cell states.
                    action, lstm_states = model.predict(
                        obs,
                        state=lstm_states,
                        episode_start=episode_starts,
                        deterministic=deterministic,
                    )
                else:
                    # Feed-forward / DQN policies only need the observation.
                    action, _ = model.predict(obs, deterministic=deterministic)
                # Render when a human window was requested.
                if render:
                    # Draw via the underlying env render path when available.
                    env.env_method("render")
                # Step the vectorized environment with the chosen action.
                obs, rewards, dones, infos = env.step(action)
                # Accumulate reward from the single eval worker.
                total_reward += float(rewards[0])
                # Capture info from the first (only) worker when present.
                if infos and isinstance(infos[0], dict):
                    # Keep the latest info for the episode summary line.
                    last_info = infos[0]
                    # Track furthest progress for the printed summary.
                    best_progress = max(
                        best_progress, float(last_info.get("progress_pct", 0.0))
                    )
                # VecEnv done flag covers terminated or truncated.
                done = bool(dones[0])
                # Feed done into the LSTM episode-start signal for the next step.
                episode_starts = dones
            # Skip incomplete episodes halted by Stop.
            if stop_event is not None and stop_event.is_set() and not done:
                # Do not record a partial episode as a finished result.
                break
            # Build a compact episode summary dict.
            summary = {
                "episode": episode + 1,
                "reward": total_reward,
                "best_progress": best_progress,
                "start_state": last_info.get("start_state"),
            }
            # Keep the summary for the return value.
            results.append(summary)
            # Print a concise episode summary.
            print(
                f"Episode {episode + 1}: reward={total_reward:.2f} "
                f"best_progress={best_progress:.1f}% "
                f"start={last_info.get('start_state')}"
            )
            # Optionally notify a UI callback on the worker thread.
            if result_callback is not None:
                # Forward the summary to the runner UI.
                result_callback(summary)
    finally:
        # Close the emulator window / process.
        env.close()
    # Return all completed episode summaries.
    return results


def main():
    """CLI entry point for evaluation runs."""
    # Build an argument parser for common eval options.
    parser = argparse.ArgumentParser(description="Run a trained Pyboy Deep Trainer agent")
    # Path to the model zip.
    parser.add_argument("--model", default="train/best_progress_model.zip")
    # Game pack id under games/.
    parser.add_argument("--game", default=None, help="Game pack id (default: mario_deluxe)")
    # Algorithm used to train the checkpoint.
    parser.add_argument(
        "--algorithm",
        default="RecurrentPPO",
        choices=["RecurrentPPO", "PPO", "QRDQN"],
    )
    # Number of evaluation episodes.
    parser.add_argument("--episodes", type=int, default=10)
    # Optional flag to disable the visible window.
    parser.add_argument("--no-render", action="store_true")
    # Optional stochastic actions for evaluation.
    parser.add_argument("--stochastic", action="store_true")
    # Device override.
    parser.add_argument("--device", default="auto")
    # Parse CLI args.
    args = parser.parse_args()
    # Launch evaluation.
    run_episodes(
        model_path=args.model,
        algorithm=args.algorithm,
        num_episodes=args.episodes,
        render=not args.no_render,
        deterministic=not args.stochastic,
        device=args.device,
        game_id=args.game,
    )


if __name__ == "__main__":
    # Run the CLI when executed as a script.
    main()
