"""Config form builder for the Mario training booth."""

import tkinter as tk
from tkinter import ttk

from ui_tooltips import TRAIN_TOOLTIPS, attach_tooltip, tip_pair


def build_config_form(
    config_frame, vars_map, on_browse, on_start, on_stop, on_gpu_preset=None
):
    """Add labeled controls and return (start_btn, stop_btn, status_var)."""

    # Helper to place a labeled control row with matching tooltips.
    def add_row(row, label, widget, tip_key=None):
        # Left label in Courier.
        label_widget = ttk.Label(config_frame, text=label, style="GB.TLabel")
        # Place the label in the grid.
        label_widget.grid(row=row, column=0, sticky="w", pady=2)
        # Right control widget.
        widget.grid(row=row, column=1, sticky="ew", pady=2)
        # Attach hover help when a tip key / text is available.
        tip_text = TRAIN_TOOLTIPS.get(tip_key or label)
        # Only wire tooltips when we have copy for this control.
        if tip_text:
            # Bind the same explanation to label and control.
            tip_pair(label_widget, widget, tip_text)
        # Return the label for optional further binding.
        return label_widget

    # Row counter for grid layout inside the config panel.
    row = 0
    # Game pack selector (one folder under games/).
    game_values = list(vars_map.get("_game_ids", ["mario_deluxe"]))
    # Prefer the list attached by the dashboard when available.
    if hasattr(vars_map["game_id"], "game_ids"):
        # Use the live catalogue from the dashboard.
        game_values = list(vars_map["game_id"].game_ids)
    # GAME row.
    add_row(
        row,
        "GAME",
        ttk.Combobox(
            config_frame,
            textvariable=vars_map["game_id"],
            values=game_values or ["mario_deluxe"],
            state="readonly",
            width=18,
            style="GB.TCombobox",
        ),
        tip_key="GAME",
    )
    row += 1
    # Algorithm selector.
    add_row(
        row,
        "ALGORITHM",
        ttk.Combobox(
            config_frame,
            textvariable=vars_map["algorithm"],
            values=["RecurrentPPO", "PPO", "QRDQN"],
            state="readonly",
            width=18,
            style="GB.TCombobox",
        ),
    )
    row += 1
    # Parallel environment count (high ceilings for GPU-friendly rollouts).
    add_row(
        row,
        "PARALLEL ENVS",
        ttk.Spinbox(
            config_frame,
            from_=1,
            to=64,
            textvariable=vars_map["n_envs"],
            width=18,
            style="GB.TSpinbox",
        ),
    )
    row += 1
    # VecEnv backend selector.
    add_row(
        row,
        "VEC ENV",
        ttk.Combobox(
            config_frame,
            textvariable=vars_map["vec_env_type"],
            values=["subproc", "dummy"],
            state="readonly",
            width=18,
            style="GB.TCombobox",
        ),
    )
    row += 1
    # Total timesteps.
    add_row(
        row,
        "TIMESTEPS",
        ttk.Entry(
            config_frame,
            textvariable=vars_map["total_timesteps"],
            width=20,
            style="GB.TEntry",
        ),
    )
    row += 1
    # Frame skip.
    add_row(
        row,
        "FRAME SKIP",
        ttk.Spinbox(
            config_frame,
            from_=1,
            to=30,
            textvariable=vars_map["frame_skip"],
            width=18,
            style="GB.TSpinbox",
        ),
    )
    row += 1
    # Learning rate.
    add_row(
        row,
        "LEARN RATE",
        ttk.Entry(
            config_frame,
            textvariable=vars_map["learning_rate"],
            width=20,
            style="GB.TEntry",
        ),
    )
    row += 1
    # Rollout n_steps.
    add_row(
        row,
        "N_STEPS",
        ttk.Entry(
            config_frame,
            textvariable=vars_map["n_steps"],
            width=20,
            style="GB.TEntry",
        ),
    )
    row += 1
    # Batch size.
    add_row(
        row,
        "BATCH",
        ttk.Entry(
            config_frame,
            textvariable=vars_map["batch_size"],
            width=20,
            style="GB.TEntry",
        ),
    )
    row += 1
    # PPO epoch count (more epochs = more GPU time per rollout).
    add_row(
        row,
        "N_EPOCHS",
        ttk.Entry(
            config_frame,
            textvariable=vars_map["n_epochs"],
            width=20,
            style="GB.TEntry",
        ),
    )
    row += 1
    # Device selector.
    add_row(
        row,
        "DEVICE",
        ttk.Combobox(
            config_frame,
            textvariable=vars_map["device"],
            values=["auto", "cuda", "cpu"],
            state="readonly",
            width=18,
            style="GB.TCombobox",
        ),
    )
    row += 1
    # How many agent panes to show.
    add_row(
        row,
        "VIS AGENTS",
        ttk.Spinbox(
            config_frame,
            from_=1,
            to=16,
            textvariable=vars_map["visual_agents"],
            width=18,
            style="GB.TSpinbox",
        ),
    )
    row += 1
    # Toggle live visuals.
    show_btn = ttk.Checkbutton(
        config_frame,
        text="SHOW LIVE SCREENS",
        variable=vars_map["show_visuals"],
        style="GB.TCheckbutton",
    )
    # Place the live-screens toggle.
    show_btn.grid(row=row, column=0, columnspan=2, sticky="w", pady=4)
    # Tooltip for live screens.
    attach_tooltip(show_btn, TRAIN_TOOLTIPS["SHOW LIVE SCREENS"])
    row += 1
    # Resume path.
    add_row(
        row,
        "RESUME",
        ttk.Entry(
            config_frame,
            textvariable=vars_map["resume_path"],
            width=20,
            style="GB.TEntry",
        ),
    )
    row += 1
    # Browse button for resume checkpoints.
    browse_btn = ttk.Button(
        config_frame, text="BROWSE .ZIP", command=on_browse, style="GB.TButton"
    )
    # Place the browse button.
    browse_btn.grid(row=row, column=0, columnspan=2, sticky="ew", pady=4)
    # Tooltip for browse.
    attach_tooltip(browse_btn, TRAIN_TOOLTIPS["BROWSE .ZIP"])
    row += 1
    # One-click GPU saturation preset for high-end cards.
    if on_gpu_preset is not None:
        # Place the preset button above Start.
        preset_btn = ttk.Button(
            config_frame,
            text="GPU PRESET (5090+)",
            command=on_gpu_preset,
            style="GB.TButton",
        )
        # Grid the preset button.
        preset_btn.grid(row=row, column=0, columnspan=2, sticky="ew", pady=4)
        # Tooltip for the GPU preset.
        attach_tooltip(preset_btn, TRAIN_TOOLTIPS["GPU PRESET (5090+)"])
        # Advance the row counter.
        row += 1
    # Start training button.
    start_btn = ttk.Button(
        config_frame, text="▶ START", command=on_start, style="Start.TButton"
    )
    # Place the start button.
    start_btn.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(10, 4))
    # Tooltip for start.
    attach_tooltip(start_btn, TRAIN_TOOLTIPS["▶ START"])
    row += 1
    # Stop training button (disabled until a run begins).
    stop_btn = ttk.Button(
        config_frame,
        text="■ STOP",
        command=on_stop,
        style="Stop.TButton",
        state=tk.DISABLED,
    )
    # Place the stop button under start.
    stop_btn.grid(row=row, column=0, columnspan=2, sticky="ew", pady=4)
    # Tooltip for stop.
    attach_tooltip(stop_btn, TRAIN_TOOLTIPS["■ STOP"])
    row += 1
    # Status label under the controls.
    status_var = tk.StringVar(value="IDLE — PRESS START TO TRAIN ACROSS .STATE FILES")
    # Status text widget.
    ttk.Label(
        config_frame, textvariable=status_var, style="GB.TLabel", wraplength=260
    ).grid(row=row, column=0, columnspan=2, sticky="w", pady=8)
    # Return the primary controls for the dashboard.
    return start_btn, stop_btn, status_var
