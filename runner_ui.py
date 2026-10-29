"""Pyboy Deep Trainer — model runner booth for testing checkpoints."""

import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from gb_theme import GB_DARKEST, GB_LIGHTEST, apply_gb_theme
from pdt import APP_NAME
from pdt.game_pack import default_game_id, list_game_ids
from run_model import list_checkpoints, run_episodes


class ModelRunnerDashboard:
    """Tkinter booth for selecting a checkpoint and rolling out episodes."""

    def __init__(self, root):
        # Keep a reference to the Tk root window.
        self.root = root
        # Product window title.
        self.root.title(f"{APP_NAME} // RUNNER")
        # Comfortable default window size.
        self.root.geometry("760x640")
        # Paint the chassis background dark olive.
        self.root.configure(bg=GB_DARKEST)
        # Background thread handle for non-blocking evaluation.
        self.run_thread = None
        # Flag flipped while an evaluation is active.
        self.is_running = False
        # Cooperative stop flag checked inside run_episodes.
        self.stop_event = threading.Event()
        # Apply the Game Boy ttk theme before building widgets.
        apply_gb_theme(self.root)
        # Build all widgets.
        self._build_layout()
        # Populate the checkpoint dropdown on launch.
        self._refresh_models()

    def _build_layout(self):
        """Create model selector, run controls, and results log."""
        # Outer dark chassis frame.
        main = ttk.Frame(self.root, style="GB.TFrame", padding=12)
        # Stretch the main frame to fill the window.
        main.pack(fill=tk.BOTH, expand=True)
        # Banner row across the top.
        banner = ttk.Frame(main, style="GB.TFrame")
        # Pack the banner full width.
        banner.pack(fill=tk.X, pady=(0, 10))
        # Big pixel-ish title.
        ttk.Label(
            banner, text=f"★ {APP_NAME.upper()} RUNNER ★", style="Title.TLabel"
        ).pack()
        # Subtitle describing the booth.
        ttk.Label(
            banner,
            text="LOAD A CHECKPOINT  ·  ROLL OUT TEST EPISODES",
            style="Sub.TLabel",
        ).pack()
        # Config panel for model selection.
        config = ttk.LabelFrame(
            main, text=" CHECKPOINT ", style="GB.TLabelframe", padding=10
        )
        # Pack the config panel.
        config.pack(fill=tk.X, pady=(0, 8))
        # Bind Tk variables for runner options.
        game_ids = list_game_ids() or ["mario_deluxe"]
        # Default game.
        default_game = default_game_id() or game_ids[0]
        # Form variables.
        self.vars = {
            "game_id": tk.StringVar(value=default_game),
            "model_path": tk.StringVar(value=""),
            "algorithm": tk.StringVar(value="RecurrentPPO"),
            "episodes": tk.IntVar(value=5),
            "device": tk.StringVar(value="auto"),
            "render": tk.BooleanVar(value=True),
            "deterministic": tk.BooleanVar(value=True),
            "checkpoint_dir": tk.StringVar(value="./train/"),
        }
        # Row counter for the form grid.
        row = 0
        # Game pack selector.
        ttk.Label(config, text="GAME", style="GB.TLabel").grid(
            row=row, column=0, sticky="w", pady=2
        )
        # Game combobox.
        ttk.Combobox(
            config,
            textvariable=self.vars["game_id"],
            values=game_ids,
            state="readonly",
            width=18,
            style="GB.TCombobox",
        ).grid(row=row, column=1, sticky="w", pady=2)
        row += 1
        # Checkpoint directory label.
        ttk.Label(config, text="TRAIN DIR", style="GB.TLabel").grid(
            row=row, column=0, sticky="w", pady=2
        )
        # Checkpoint directory entry.
        ttk.Entry(
            config, textvariable=self.vars["checkpoint_dir"], width=42, style="GB.TEntry"
        ).grid(row=row, column=1, sticky="ew", pady=2)
        # Refresh button to re-scan the train folder.
        ttk.Button(
            config, text="REFRESH", command=self._refresh_models, style="GB.TButton"
        ).grid(row=row, column=2, padx=4, pady=2)
        row += 1
        # Model dropdown label.
        ttk.Label(config, text="MODEL", style="GB.TLabel").grid(
            row=row, column=0, sticky="w", pady=2
        )
        # Combobox listing discovered checkpoints.
        self.model_combo = ttk.Combobox(
            config,
            textvariable=self.vars["model_path"],
            values=[],
            width=40,
            style="GB.TCombobox",
        )
        # Place the model combo.
        self.model_combo.grid(row=row, column=1, sticky="ew", pady=2)
        # Browse for an arbitrary zip outside train/.
        ttk.Button(
            config, text="BROWSE", command=self._browse_model, style="GB.TButton"
        ).grid(row=row, column=2, padx=4, pady=2)
        row += 1
        # Algorithm selector.
        ttk.Label(config, text="ALGORITHM", style="GB.TLabel").grid(
            row=row, column=0, sticky="w", pady=2
        )
        # Algorithm combobox.
        ttk.Combobox(
            config,
            textvariable=self.vars["algorithm"],
            values=["RecurrentPPO", "PPO", "QRDQN"],
            state="readonly",
            width=18,
            style="GB.TCombobox",
        ).grid(row=row, column=1, sticky="w", pady=2)
        row += 1
        # Episode count.
        ttk.Label(config, text="EPISODES", style="GB.TLabel").grid(
            row=row, column=0, sticky="w", pady=2
        )
        # Episode spinbox.
        ttk.Spinbox(
            config,
            from_=1,
            to=100,
            textvariable=self.vars["episodes"],
            width=18,
            style="GB.TSpinbox",
        ).grid(row=row, column=1, sticky="w", pady=2)
        row += 1
        # Device selector.
        ttk.Label(config, text="DEVICE", style="GB.TLabel").grid(
            row=row, column=0, sticky="w", pady=2
        )
        # Device combobox.
        ttk.Combobox(
            config,
            textvariable=self.vars["device"],
            values=["auto", "cuda", "cpu"],
            state="readonly",
            width=18,
            style="GB.TCombobox",
        ).grid(row=row, column=1, sticky="w", pady=2)
        row += 1
        # Render toggle.
        ttk.Checkbutton(
            config,
            text="SHOW GAME WINDOW",
            variable=self.vars["render"],
            style="GB.TCheckbutton",
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=2)
        row += 1
        # Deterministic action toggle.
        ttk.Checkbutton(
            config,
            text="DETERMINISTIC ACTIONS",
            variable=self.vars["deterministic"],
            style="GB.TCheckbutton",
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=2)
        # Stretch the middle column.
        config.columnconfigure(1, weight=1)
        # Button row under the config panel.
        buttons = ttk.Frame(main, style="GB.TFrame")
        # Pack the button row.
        buttons.pack(fill=tk.X, pady=8)
        # Start evaluation button.
        self.start_btn = ttk.Button(
            buttons, text="▶ RUN MODEL", command=self._start_run, style="Start.TButton"
        )
        # Pack Start on the left.
        self.start_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        # Stop evaluation button.
        self.stop_btn = ttk.Button(
            buttons,
            text="■ STOP",
            command=self._stop_run,
            style="Stop.TButton",
            state=tk.DISABLED,
        )
        # Pack Stop beside Start.
        self.stop_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))
        # Status line under the buttons.
        self.status_var = tk.StringVar(value="IDLE — SELECT A CHECKPOINT AND PRESS RUN")
        # Status label.
        ttk.Label(
            main, textvariable=self.status_var, style="Sub.TLabel", wraplength=700
        ).pack(fill=tk.X, pady=(0, 8))
        # Results log frame.
        log_frame = ttk.LabelFrame(
            main, text=" EPISODE RESULTS ", style="GB.TLabelframe", padding=8
        )
        # Let the log expand.
        log_frame.pack(fill=tk.BOTH, expand=True)
        # Text widget for episode summaries.
        self.log = tk.Text(
            log_frame,
            height=18,
            wrap=tk.WORD,
            bg=GB_LIGHTEST,
            fg=GB_DARKEST,
            insertbackground=GB_DARKEST,
            font=("Courier New", 9),
            relief=tk.SOLID,
            borderwidth=4,
            highlightthickness=0,
        )
        # Pack the log widget.
        self.log.pack(fill=tk.BOTH, expand=True)
        # Seed a short help line.
        self.log.insert(
            tk.END,
            "> Pick a model from ./train/ or browse to a .zip\n"
            "> Algorithm must match how the checkpoint was trained\n"
            "> Deterministic=ON uses the greedy action each step\n",
        )

    def _refresh_models(self):
        """Rescan the train directory and refresh the model dropdown."""
        # Read the configured checkpoint directory.
        checkpoint_dir = self.vars["checkpoint_dir"].get()
        # Discover zip / bare checkpoints.
        paths = list_checkpoints(checkpoint_dir)
        # Update the combobox values.
        self.model_combo["values"] = paths
        # Auto-select the newest checkpoint when available.
        if paths and not self.vars["model_path"].get():
            # Prefer the most recently modified model.
            self.vars["model_path"].set(paths[0])
        # Update status with the discovery count.
        self.status_var.set(f"FOUND {len(paths)} CHECKPOINT(S) IN {checkpoint_dir}")

    def _browse_model(self):
        """Open a file dialog to pick any model zip."""
        # Ask the user for a checkpoint path.
        path = filedialog.askopenfilename(
            title="Select model checkpoint",
            filetypes=[("SB3 model", "*.zip"), ("All files", "*.*")],
        )
        # Update the model path when a file was chosen.
        if path:
            # Store the selected path.
            self.vars["model_path"].set(path)

    def _set_running_controls(self, running):
        """Enable/disable Start and Stop to match the current run state."""
        # While running: Start off, Stop on.
        if running:
            # Disable Start so duplicate runs cannot spawn.
            self.start_btn.config(state=tk.DISABLED)
            # Enable Stop so the user can halt cooperatively.
            self.stop_btn.config(state=tk.NORMAL)
        else:
            # Re-enable Start after the run ends.
            self.start_btn.config(state=tk.NORMAL)
            # Disable Stop when idle.
            self.stop_btn.config(state=tk.DISABLED)

    def _start_run(self):
        """Launch evaluation on a background thread."""
        # Ignore duplicate clicks while a run is already active.
        if self.is_running:
            # Warn the user that evaluation is already running.
            messagebox.showinfo("Runner", "A model run is already in progress.")
            return
        # Read the selected model path.
        model_path = self.vars["model_path"].get().strip()
        # Require a model before starting.
        if not model_path or not os.path.exists(model_path):
            # Show a clear missing-file dialog.
            messagebox.showerror("Runner", f"Model not found:\n{model_path}")
            return
        # Clear any previous stop request.
        self.stop_event.clear()
        # Mark evaluation as active.
        self.is_running = True
        # Flip Start/Stop button states.
        self._set_running_controls(True)
        # Update the status line.
        self.status_var.set(
            f"RUNNING {os.path.basename(model_path)} · "
            f"{self.vars['algorithm'].get()} · "
            f"{int(self.vars['episodes'].get())} EP"
        )
        # Snapshot options for the worker thread.
        algorithm = self.vars["algorithm"].get()
        # Episode count.
        episodes = int(self.vars["episodes"].get())
        # Device string.
        device = self.vars["device"].get()
        # Whether to open the game window.
        render = bool(self.vars["render"].get())
        # Whether to use greedy actions.
        deterministic = bool(self.vars["deterministic"].get())
        # Selected game pack id.
        game_id = self.vars["game_id"].get()

        # Define the worker target that runs inside the thread.
        def _worker():
            # Best-effort evaluation; surface failures in the UI.
            try:
                # Roll out episodes with cooperative stop support.
                results = run_episodes(
                    model_path=model_path,
                    algorithm=algorithm,
                    num_episodes=episodes,
                    render=render,
                    deterministic=deterministic,
                    device=device,
                    game_id=game_id,
                    stop_event=self.stop_event,
                    result_callback=lambda summary: self.root.after(
                        0, lambda s=summary: self._on_episode(s)
                    ),
                )
                # Detect whether the stop event caused an early exit.
                was_stopped = self.stop_event.is_set()
                # Schedule a completion update on the Tk thread.
                self.root.after(
                    0, lambda: self._on_run_done(True, results, was_stopped)
                )
            except Exception as exc:
                # Schedule a failure update on the Tk thread.
                self.root.after(
                    0, lambda: self._on_run_done(False, str(exc), False)
                )

        # Start the background evaluation thread.
        self.run_thread = threading.Thread(target=_worker, daemon=True)
        # Begin evaluation.
        self.run_thread.start()

    def _stop_run(self):
        """Request a cooperative stop of the current evaluation."""
        # Ignore Stop clicks when nothing is running.
        if not self.is_running:
            # Nothing to stop.
            return
        # Signal the evaluation loop to halt.
        self.stop_event.set()
        # Update status so the user sees the request was received.
        self.status_var.set("STOPPING — WAITING FOR CURRENT EPISODE TO EXIT...")
        # Disable Stop to prevent repeated clicks while draining.
        self.stop_btn.config(state=tk.DISABLED)

    def _on_episode(self, summary):
        """Append one finished episode summary to the results log."""
        # Format a compact mono line for the log.
        line = (
            f"> EP {summary['episode']:02d}  "
            f"R={summary['reward']:.2f}  "
            f"P={summary['best_progress']:.1f}%  "
            f"{summary.get('start_state')}\n"
        )
        # Insert into the scrolling log.
        self.log.insert(tk.END, line)
        # Autoscroll to the latest line.
        self.log.see(tk.END)

    def _on_run_done(self, success, detail, was_stopped):
        """Update the UI when the background evaluation thread finishes."""
        # Clear the running flag.
        self.is_running = False
        # Restore Start / disable Stop.
        self._set_running_controls(False)
        # Clear the stop event for the next run.
        self.stop_event.clear()
        # Branch on success vs failure vs user stop.
        if success and was_stopped:
            # Show that evaluation was halted by the user.
            count = len(detail) if isinstance(detail, list) else 0
            # Status reflects a partial run.
            self.status_var.set(f"STOPPED AFTER {count} EPISODE(S)")
            # Append a stop line to the log.
            self.log.insert(tk.END, f"> Stopped after {count} episode(s)\n")
        elif success:
            # Summarize completed episodes.
            count = len(detail) if isinstance(detail, list) else 0
            # Mean reward across completed episodes when available.
            if count:
                # Compute mean return for a quick readout.
                mean_r = sum(item["reward"] for item in detail) / count
                # Compute mean best progress.
                mean_p = sum(item["best_progress"] for item in detail) / count
                # Status shows aggregate stats.
                self.status_var.set(
                    f"DONE — {count} EP  meanR={mean_r:.1f}  meanP={mean_p:.1f}%"
                )
                # Append a summary line to the log.
                self.log.insert(
                    tk.END,
                    f"> Done: {count} ep  meanR={mean_r:.2f}  meanP={mean_p:.1f}%\n",
                )
            else:
                # No episodes completed.
                self.status_var.set("DONE — NO EPISODES COMPLETED")
        else:
            # Show the error message.
            self.status_var.set("ERROR — SEE LOG")
            # Append the exception text.
            self.log.insert(tk.END, f"> ERROR: {detail}\n")
            # Pop a dialog so the failure is obvious.
            messagebox.showerror("Runner failed", detail)
        # Autoscroll after the final line.
        self.log.see(tk.END)


def main():
    """Entry point for the model runner dashboard."""
    # Create the root Tk window.
    root = tk.Tk()
    # Build the dashboard against the root window.
    ModelRunnerDashboard(root)
    # Enter the Tk event loop.
    root.mainloop()


if __name__ == "__main__":
    # Launch the runner when executed as a script.
    main()
