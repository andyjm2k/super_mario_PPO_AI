"""SB3 logger writer that forwards dump tables to the training UI queue."""

from stable_baselines3.common.logger import KVWriter


class QueueStatsWriter(KVWriter):
    """Push each logger.dump() key/value table onto a multiprocessing queue."""

    def __init__(self, stats_queue):
        # Queue shared with the Tk training booth.
        self.stats_queue = stats_queue

    def write(self, key_values, key_excluded, step=0):
        # Build a plain dict the UI can render without SB3 types.
        stats = {}
        # Copy every numeric / string metric from this dump.
        for key, value in key_values.items():
            # Skip keys excluded from the human console table when possible.
            excluded = key_excluded.get(key) if key_excluded else None
            # Mirror stdout: keep keys that appear in the human format.
            if excluded is not None and "stdout" in excluded:
                # Omit tensorboard-only keys from the booth HUD.
                continue
            # Coerce tensors / numpy scalars into plain Python numbers.
            if hasattr(value, "item"):
                try:
                    # Unwrap 0-d tensors and numpy scalars.
                    value = value.item()
                except Exception:
                    # Keep the original value if unwrap fails.
                    pass
            # Store the cleaned value under its SB3 logger key.
            stats[str(key)] = value
        # Nothing to send when the dump was empty.
        if not stats:
            # Exit quietly.
            return
        # Best-effort non-blocking put so a stalled UI cannot freeze training.
        try:
            # Payload type lets the UI multiplex frames vs stats.
            self.stats_queue.put_nowait(
                {"type": "stats", "step": int(step), "stats": stats}
            )
        except Exception:
            # Drop the update if the queue is full.
            pass

    def close(self):
        # No file handles to release for a queue-backed writer.
        return None


def format_stat_value(key, value):
    """Format a logger value for the pixel HUD cells."""
    # Non-numeric values are shown as plain text.
    if not isinstance(value, (int, float)):
        # Stringify anything unexpected.
        return str(value)
    # Integer-like counters stay as whole numbers.
    if key in {
        "time/fps",
        "time/iterations",
        "time/time_elapsed",
        "time/total_timesteps",
        "train/n_updates",
        "rollout/ep_len_mean",
    }:
        # Round means that are nearly integers for readability.
        return str(int(round(float(value))))
    # Learning rate needs more precision.
    if key == "train/learning_rate":
        # Scientific-ish fixed format matching SB3 stdout.
        return f"{float(value):.5g}"
    # Small float losses / KL terms.
    if abs(float(value)) < 0.01 and float(value) != 0.0:
        # Compact scientific style for tiny magnitudes.
        return f"{float(value):.4g}"
    # Default two-decimal display for rewards and losses.
    return f"{float(value):.3g}"
