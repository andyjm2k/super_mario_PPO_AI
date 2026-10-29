"""Pixel-art training stats board for the GBC booth UI."""

import tkinter as tk
from tkinter import ttk

from gb_theme import GB_DARKEST, GB_LIGHTEST
from stats_writer import format_stat_value


# Ordered keys shown under the ROLL column.
ROLL_KEYS = (
    "rollout/ep_len_mean",
    "rollout/ep_rew_mean",
    "rollout/success_rate",
)
# Ordered keys shown under the TIME column.
TIME_KEYS = (
    "time/fps",
    "time/iterations",
    "time/time_elapsed",
    "time/total_timesteps",
)
# Ordered keys shown under the TRAIN column.
TRAIN_KEYS = (
    "train/approx_kl",
    "train/clip_fraction",
    "train/clip_range",
    "train/entropy_loss",
    "train/explained_variance",
    "train/learning_rate",
    "train/loss",
    "train/n_updates",
    "train/policy_gradient_loss",
    "train/value_loss",
)


def _short_label(key):
    """Strip the SB3 group prefix for compact HUD labels."""
    # Split on the first slash to drop rollout/time/train.
    if "/" in key:
        # Keep only the metric name.
        return key.split("/", 1)[1]
    # Fallback to the raw key.
    return key


class StatsBoard:
    """Three-column olive HUD that mirrors SB3's stdout training table."""

    def __init__(self, parent):
        # Outer labeled frame matching the booth chrome.
        self.frame = ttk.LabelFrame(
            parent, text=" TRAIN TELEMETRY ", style="GB.TLabelframe", padding=8
        )
        # Event ticker for checkpoint / best-progress lines.
        self.event_var = tk.StringVar(value="WAITING FOR FIRST ROLLOUT DUMP...")
        # Ticker label across the top of the board.
        ttk.Label(
            self.frame, textvariable=self.event_var, style="Caption.TLabel"
        ).pack(fill=tk.X, pady=(0, 6))
        # Row that holds ROLL / TIME / TRAIN panels.
        columns = ttk.Frame(self.frame, style="Panel.TFrame")
        # Stretch the column row.
        columns.pack(fill=tk.BOTH, expand=True)
        # Map logger keys to StringVars for live updates.
        self.value_vars = {}
        # Build each themed column.
        self._build_column(columns, "ROLL", ROLL_KEYS, 0)
        # TIME column in the middle.
        self._build_column(columns, "TIME", TIME_KEYS, 1)
        # TRAIN column on the right (widest metric set).
        self._build_column(columns, "TRAIN", TRAIN_KEYS, 2)
        # Allow columns to share horizontal space evenly.
        for index in range(3):
            # Equal weight so panels stay balanced.
            columns.columnconfigure(index, weight=1)

    def _build_column(self, parent, title, keys, column_index):
        """Create one metric group panel with labeled value cells."""
        # Cream panel with a thick olive border look via LabelFrame.
        panel = ttk.LabelFrame(
            parent, text=f" {title} ", style="GB.TLabelframe", padding=6
        )
        # Place the panel in the column grid.
        panel.grid(row=0, column=column_index, sticky="nsew", padx=4, pady=2)
        # Add one row per metric key.
        for row_index, key in enumerate(keys):
            # Human-readable short label on the left.
            ttk.Label(panel, text=_short_label(key), style="GB.TLabel").grid(
                row=row_index, column=0, sticky="w", padx=(0, 8), pady=1
            )
            # Live value variable for this key.
            value_var = tk.StringVar(value="--")
            # Remember the var so update_stats can refresh it.
            self.value_vars[key] = value_var
            # Value cell on the right in a darker “screen” label.
            value_label = tk.Label(
                panel,
                textvariable=value_var,
                bg=GB_LIGHTEST,
                fg=GB_DARKEST,
                font=("Courier New", 9, "bold"),
                relief=tk.SOLID,
                bd=2,
                padx=4,
                pady=1,
                width=12,
                anchor="e",
            )
            # Dock the value cell to the right edge of the panel.
            value_label.grid(row=row_index, column=1, sticky="e", pady=1)
        # Stretch the value column inside the panel.
        panel.columnconfigure(1, weight=1)

    def pack(self, **kwargs):
        """Pack the outer frame into the parent layout."""
        # Forward pack options to the outer LabelFrame.
        self.frame.pack(**kwargs)

    def set_event(self, message):
        """Update the checkpoint / progress event ticker."""
        # Prefix with a retro prompt glyph.
        self.event_var.set(f"> {message}")

    def update_stats(self, stats):
        """Refresh HUD cells from an SB3 logger dump dict."""
        # Guard against empty payloads.
        if not isinstance(stats, dict):
            # Nothing to render.
            return
        # Update every known key that appears in this dump.
        for key, value_var in self.value_vars.items():
            # Skip keys the dump has not produced yet.
            if key not in stats:
                # Leave the previous value on screen.
                continue
            # Format and push the new value into the cell.
            value_var.set(format_stat_value(key, stats[key]))
        # Surface total timesteps in the event line when present.
        if "time/total_timesteps" in stats:
            # Compact step counter next to the last event context.
            steps = format_stat_value(
                "time/total_timesteps", stats["time/total_timesteps"]
            )
            # Append step count without wiping a richer checkpoint message.
            current = self.event_var.get()
            # Only auto-rewrite when still on the waiting placeholder.
            if "WAITING" in current or "timesteps=" in current:
                # Show the latest step count as the default ticker.
                self.event_var.set(f"> DUMP @ timesteps={steps}")
