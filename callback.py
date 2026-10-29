"""Checkpoint and metrics callbacks used during Mario training."""

import os

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from stats_writer import QueueStatsWriter


class TrainAndLoggingCallback(BaseCallback):
    """Save periodic checkpoints, stream UI frames, and honor stop events."""

    def __init__(
        self,
        check_freq,
        save_path,
        verbose=1,
        frame_queue=None,
        stop_event=None,
    ):
        # Initialize the SB3 callback base class.
        super().__init__(verbose)
        # How often (in callback steps) to write a checkpoint.
        self.check_freq = check_freq
        # Directory where model zip files are written.
        self.save_path = save_path
        # Optional multiprocessing queue for live UI frames.
        self.frame_queue = frame_queue
        # Optional threading/multiprocessing Event set by the UI to halt.
        self.stop_event = stop_event
        # Best mean progress observed so far across rollouts.
        self.best_mean_progress = -np.inf
        # True when the last stop was requested by the user / UI.
        self.stopped_by_user = False

    def _init_callback(self):
        """Create the checkpoint directory on the first training step."""
        # Ensure the save directory exists before writing models.
        if self.save_path:
            # Create nested directories as needed.
            os.makedirs(self.save_path, exist_ok=True)

    def _on_training_start(self):
        """Hook the UI stats writer once SB3 has created the logger."""
        # Skip when no UI queue is attached.
        if self.frame_queue is None:
            # Nothing to stream.
            return
        # Avoid registering the writer twice if learn() is called again.
        already = any(
            isinstance(fmt, QueueStatsWriter) for fmt in self.logger.output_formats
        )
        # Only append when this callback has not already installed one.
        if not already:
            # Forward every logger.dump() table into the booth queue.
            self.logger.output_formats.append(QueueStatsWriter(self.frame_queue))

    def _on_step(self):
        """Save checkpoints, stream frames, and stop when requested."""
        # Cooperative stop: returning False ends model.learn() cleanly.
        if self.stop_event is not None and self.stop_event.is_set():
            # Remember that the user requested the halt.
            self.stopped_by_user = True
            # Persist a stop snapshot so progress is not lost.
            stop_path = os.path.join(self.save_path, "stopped_model")
            # Best-effort save; never block shutdown on IO errors.
            try:
                # Write the current weights before exiting.
                self.model.save(stop_path)
                # Log the stop checkpoint when verbose.
                if self.verbose:
                    # Print where the interrupted model was saved.
                    print(f"Stop requested — saved {stop_path}")
            except Exception as exc:
                # Surface save failures without preventing shutdown.
                if self.verbose:
                    # Print the save error for debugging.
                    print(f"Stop save failed: {exc}")
            # Returning False tells SB3 to end training.
            return False
        # Periodically serialize the current policy weights.
        if self.n_calls % self.check_freq == 0:
            # Build a unique checkpoint path from the step counter.
            model_path = os.path.join(self.save_path, f"best_model_{self.n_calls}")
            # Save the SB3 / sb3-contrib model zip.
            self.model.save(model_path)
            # Build the human-readable checkpoint message.
            checkpoint_msg = f"Saved checkpoint: {model_path}"
            # Log when verbosity is enabled.
            if self.verbose:
                # Print a short checkpoint notice.
                print(checkpoint_msg)
            # Mirror the checkpoint line into the training booth ticker.
            self._push_event(checkpoint_msg)
            # Attempt to track mean progress from the latest infos.
            self._maybe_save_best_progress()
        # Push live frames to the UI when a queue was provided.
        if self.frame_queue is not None and self.n_calls % 8 == 0:
            # Best-effort frame capture; never break training on UI errors.
            try:
                # Ask each subprocess env for its cached RGB frame.
                frames = self.training_env.get_attr("last_frame")
                # Ask each env for the latest info dict.
                infos = self.training_env.get_attr("last_info")
                # Non-blocking put so a stalled UI cannot freeze training.
                self.frame_queue.put_nowait(
                    {"type": "frames", "frames": frames, "infos": infos}
                )
            except Exception:
                # Ignore full queues / serialization edge cases.
                pass
        # Continue training unless a stop was requested above.
        return True

    def _push_event(self, message):
        """Send a checkpoint / progress event string to the UI queue."""
        # Skip when no UI queue is attached.
        if self.frame_queue is None:
            # Nothing to do.
            return
        # Best-effort event push for the booth ticker.
        try:
            # Event payloads are consumed by the StatsBoard ticker.
            self.frame_queue.put_nowait({"type": "event", "message": str(message)})
        except Exception:
            # Drop the event if the queue is full.
            pass

    def _maybe_save_best_progress(self):
        """Save an extra checkpoint when mean progress improves."""
        # Pull the latest info dicts from the vectorized environments.
        try:
            # Each worker exposes last_info with progress_pct when available.
            infos = self.training_env.get_attr("last_info")
        except Exception:
            # Skip quietly if workers cannot be queried.
            return
        # Collect progress values from workers that have reported any.
        progresses = [
            float(info.get("progress_pct", 0.0))
            for info in infos
            if isinstance(info, dict)
        ]
        # Nothing to do when no progress data is available yet.
        if not progresses:
            # Exit early.
            return
        # Compute the mean progress across parallel agents.
        mean_progress = float(np.mean(progresses))
        # Only keep a dedicated best checkpoint when progress improves.
        if mean_progress > self.best_mean_progress:
            # Update the tracked best mean progress.
            self.best_mean_progress = mean_progress
            # Write a stable filename for the current best policy.
            best_path = os.path.join(self.save_path, "best_progress_model")
            # Persist the improved model.
            self.model.save(best_path)
            # Build the human-readable best-progress message.
            progress_msg = (
                f"New best mean progress {mean_progress:.2f}% -> {best_path}"
            )
            # Announce the improvement when verbose.
            if self.verbose:
                # Print the new best mean progress.
                print(progress_msg)
            # Mirror the best-progress line into the training booth ticker.
            self._push_event(progress_msg)
