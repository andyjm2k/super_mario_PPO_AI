"""Pyboy Deep Trainer — multi-agent training booth with declarative rewards."""

import multiprocessing as mp
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from gb_theme import GB_DARKEST, GB_LIGHTEST, apply_gb_theme
from gpu_tune import suggest_gpu_hyperparams, validate_batch_divides_buffer
from model_architecture import default_policy_for_algorithm
from observation_pipeline import resolve_frame_stack
from pdt import APP_NAME
from pdt.game_pack import default_game_id, list_game_ids, load_game_pack
from train_model import train_model
from training_config import TrainingConfig
from ui_arch_form import build_arch_form
from ui_config_form import build_config_form
from ui_previews import rebuild_preview_grid, update_previews
from ui_reward_form import RewardFormController, build_reward_form
from ui_stats_panel import StatsBoard


class TrainingDashboard:
    """Tkinter booth for configuring and monitoring parallel PyBoy agents."""

    def __init__(self, root):
        # Keep a reference to the Tk root window.
        self.root = root
        # Product window title.
        self.root.title(f"{APP_NAME} // TRAIN")
        # Start with a taller window so the telemetry board fits.
        self.root.geometry("1180x900")
        # Paint the chassis background dark olive.
        self.root.configure(bg=GB_DARKEST)
        # Multiprocessing queue used to stream frames, stats, and events.
        self.frame_queue = mp.Queue(maxsize=64)
        # Background thread handle for non-blocking training.
        self.train_thread = None
        # Flag flipped when training finishes or fails.
        self.is_training = False
        # Cooperative stop flag checked by TrainAndLoggingCallback.
        self.stop_event = threading.Event()
        # PhotoImage references must be kept alive for Tk.
        self.photo_refs = []
        # Apply the Game Boy ttk theme before building widgets.
        apply_gb_theme(self.root)
        # Build all widgets.
        self._build_layout()
        # Load the default game pack into the rewards editor.
        self._on_game_changed()
        # Start polling the frame queue for live previews.
        self.root.after(100, self._poll_frames)

    def _build_layout(self):
        """Create configuration controls, status text, and the agent grid."""
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
            banner, text=f"★ {APP_NAME.upper()} ★", style="Title.TLabel"
        ).pack()
        # Subtitle describing the booth.
        ttk.Label(
            banner,
            text="GENERIC PYBOY RL  ·  PER-GAME PACKS  ·  DECLARATIVE REWARDS",
            style="Sub.TLabel",
        ).pack()
        # Body holds config (left) and screens (right).
        body = ttk.Frame(main, style="GB.TFrame")
        # Pack the body so it expands.
        body.pack(fill=tk.BOTH, expand=True)
        # Left column holds a notebook with train / model / rewards tabs.
        left = ttk.Frame(body, style="GB.TFrame")
        # Dock the left notebook column.
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        # Notebook switches between parallel config, arch, and rewards.
        notebook = ttk.Notebook(left)
        # Pack the notebook so tabs fill the column.
        notebook.pack(fill=tk.BOTH, expand=True)
        # Tab 1: parallelization / training hyper-parameters.
        config_frame = ttk.Frame(notebook, style="Panel.TFrame", padding=10)
        # Tab 2: neural network architecture knobs.
        arch_frame = ttk.Frame(notebook, style="Panel.TFrame", padding=10)
        # Tab 3: declarative reward editor + memory import.
        reward_frame = ttk.Frame(notebook, style="Panel.TFrame", padding=10)
        # Register tabs with olive captions.
        notebook.add(config_frame, text=" TRAIN ")
        # Architecture tab for layers / neurons / LSTM.
        notebook.add(arch_frame, text=" MODEL ")
        # Rewards tab for declarative rules.
        notebook.add(reward_frame, text=" REWARDS ")
        # Right column holds live agent visuals and status.
        right = ttk.Frame(body, style="GB.TFrame")
        # Dock the preview/status panel to the right and let it expand.
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        # Discover installed game packs.
        game_ids = list_game_ids() or ["mario_deluxe"]
        # Default game id.
        default_game = default_game_id() or game_ids[0]
        # Bind Tk variables for every configurable field.
        self.vars = {
            "game_id": tk.StringVar(value=default_game),
            "games_root": tk.StringVar(value="games"),
            "rewards_path": tk.StringVar(value=""),
            "memory_path": tk.StringVar(value=""),
            "algorithm": tk.StringVar(value="RecurrentPPO"),
            "n_envs": tk.IntVar(value=16),
            "total_timesteps": tk.IntVar(value=2_000_000),
            "frame_skip": tk.IntVar(value=4),
            "learning_rate": tk.DoubleVar(value=0.00025),
            "n_steps": tk.IntVar(value=1024),
            "batch_size": tk.IntVar(value=2048),
            "n_epochs": tk.IntVar(value=8),
            "vec_env_type": tk.StringVar(value="subproc"),
            "visual_agents": tk.IntVar(value=4),
            "show_visuals": tk.BooleanVar(value=True),
            "device": tk.StringVar(value="auto"),
            "resume_path": tk.StringVar(value=""),
            "checkpoint_dir": tk.StringVar(value="./train/"),
            "log_dir": tk.StringVar(value="./logs/"),
            "pi_arch": tk.StringVar(value="512,512"),
            "vf_arch": tk.StringVar(value="512,512"),
            "q_arch": tk.StringVar(value="512,512"),
            "features_dim": tk.IntVar(value=512),
            "lstm_hidden_size": tk.IntVar(value=512),
            "n_lstm_layers": tk.IntVar(value=1),
            "shared_lstm": tk.BooleanVar(value=False),
            "enable_critic_lstm": tk.BooleanVar(value=True),
            "activation": tk.StringVar(value="relu"),
            "ortho_init": tk.BooleanVar(value=True),
            "_game_ids": game_ids,
        }
        # Attach catalogue onto the StringVar for the config form helper.
        self.vars["game_id"].game_ids = game_ids
        # Reload pack assets when the game selection changes.
        self.vars["game_id"].trace_add("write", lambda *_: self._on_game_changed())
        # Populate the train-tab form controls.
        self.start_btn, self.stop_btn, self.status_var = build_config_form(
            config_frame,
            self.vars,
            self._browse_resume,
            self._start_training,
            self._stop_training,
            on_gpu_preset=self._apply_gpu_preset,
        )
        # Populate the architecture-tab controls.
        build_arch_form(arch_frame, self.vars)
        # Rewards editor controller shared with the REWARDS tab.
        self.reward_controller = RewardFormController(
            self.root,
            self.vars,
            on_status=self._append_metric,
        )
        # Build the rewards editor widgets.
        build_reward_form(reward_frame, self.reward_controller)
        # Live preview section title.
        preview = ttk.LabelFrame(
            right, text=" LIVE AGENT SCREENS ", style="GB.TLabelframe", padding=8
        )
        # Let the preview area expand.
        preview.pack(fill=tk.BOTH, expand=True)
        # Container for the grid of agent canvases.
        self.grid_frame = ttk.Frame(preview, style="Panel.TFrame")
        # Pack the grid container.
        self.grid_frame.pack(fill=tk.BOTH, expand=True)
        # Pixel HUD mirroring SB3 rollout / time / train dumps.
        self.stats_board = StatsBoard(right)
        # Place telemetry under the live screens.
        self.stats_board.pack(fill=tk.BOTH, expand=False, pady=(8, 0))
        # Compact event / tip log under the telemetry board.
        self.metrics = tk.Text(
            right,
            height=5,
            wrap=tk.WORD,
            bg=GB_LIGHTEST,
            fg=GB_DARKEST,
            insertbackground=GB_DARKEST,
            font=("Courier New", 9),
            relief=tk.SOLID,
            borderwidth=4,
            highlightthickness=0,
        )
        # Pack the metrics panel at the bottom.
        self.metrics.pack(fill=tk.X, pady=(8, 0))
        # Seed an introductory help line.
        self.metrics.insert(
            tk.END,
            f"> {APP_NAME}: pick a GAME pack, tune REWARDS, then TRAIN\n"
            "> IMPORT MEMORY on the REWARDS tab to add address maps\n"
            "> ROMs/states live under games/<id>/ so titles stay separate\n",
        )
        # Build the initial empty preview panes.
        (
            self.preview_labels,
            self.caption_labels,
            self.photo_refs,
        ) = rebuild_preview_grid(self.grid_frame, self.vars["visual_agents"].get())

    def _append_metric(self, message):
        """Append one status line to the scrolling metrics pane."""
        # Insert the message with a prompt prefix.
        self.metrics.insert(tk.END, f"> {message}\n")
        # Autoscroll.
        self.metrics.see(tk.END)

    def _on_game_changed(self):
        """Reload memory/rewards editors when the selected game pack changes."""
        # Selected pack id.
        game_id = self.vars["game_id"].get()
        # Games root folder.
        games_root = self.vars["games_root"].get()
        try:
            # Resolve the pack from disk.
            pack = load_game_pack(game_id, games_root=games_root)
            # Push pack assets into the rewards editor.
            self.reward_controller.load_from_pack(pack)
            # Update status strip with pack name + state count.
            self.status_var.set(
                f"GAME {pack.name} · {len(pack.discover_states())} STATES"
            )
        except Exception as exc:
            # Surface pack load errors without crashing the UI.
            self._append_metric(f"Game pack error: {exc}")

    def _rebuild_preview_grid(self, count):
        """Recreate the agent preview labels for the requested pane count."""
        # Delegate grid construction to the shared preview helper.
        (
            self.preview_labels,
            self.caption_labels,
            self.photo_refs,
        ) = rebuild_preview_grid(self.grid_frame, count)

    def _browse_resume(self):
        """Open a file dialog to pick a model zip to resume from."""
        # Ask the user for a checkpoint path.
        path = filedialog.askopenfilename(
            title="Select model checkpoint",
            filetypes=[("SB3 model", "*.zip"), ("All files", "*.*")],
        )
        # Update the resume path variable when a file was chosen.
        if path:
            # Store the selected path.
            self.vars["resume_path"].set(path)

    def _apply_gpu_preset(self):
        """Fill the form with large-batch settings for high-end NVIDIA GPUs."""
        # Build a preset matched to the currently selected algorithm.
        preset = suggest_gpu_hyperparams(self.vars["algorithm"].get())
        # Apply each preset field onto the matching Tk variable.
        self.vars["n_envs"].set(preset["n_envs"])
        # Longer rollouts between updates.
        self.vars["n_steps"].set(preset["n_steps"])
        # Larger minibatches for heavier train() kernels.
        self.vars["batch_size"].set(preset["batch_size"])
        # More optimization epochs per rollout.
        self.vars["n_epochs"].set(preset["n_epochs"])
        # Force CUDA for the preset.
        self.vars["device"].set(preset["device"])
        # Disable live screens so CPU cores stay on PyBoy workers.
        self.vars["show_visuals"].set(preset["show_visuals"])
        # Fewer preview panes if visuals are re-enabled later.
        self.vars["visual_agents"].set(preset["visual_agents"])
        # Confirm in the status line.
        self.status_var.set(
            f"GPU PRESET LOADED — {preset['n_envs']} envs · "
            f"batch {preset['batch_size']} · {preset['n_epochs']} epochs"
        )
        # Also log the change in the scrolling pane.
        self.metrics.insert(
            tk.END,
            f"> GPU preset: envs={preset['n_envs']} steps={preset['n_steps']} "
            f"batch={preset['batch_size']} epochs={preset['n_epochs']} "
            f"visuals={preset['show_visuals']}\n",
        )
        # Autoscroll the log.
        self.metrics.see(tk.END)

    def _build_config(self):
        """Materialize a TrainingConfig from the UI controls."""
        # Read the selected algorithm for policy / stack defaults.
        algorithm = self.vars["algorithm"].get()
        # Read numeric hyper-parameters once for validation + config.
        n_envs = int(self.vars["n_envs"].get())
        # Rollout length per env.
        n_steps = int(self.vars["n_steps"].get())
        # Minibatch size for train().
        batch_size = int(self.vars["batch_size"].get())
        # Ensure PPO buffer geometry is valid before spawning workers.
        error = validate_batch_divides_buffer(n_envs, n_steps, batch_size, algorithm)
        # Abort config build when the geometry is invalid.
        if error:
            # Surface the problem to the user immediately.
            raise ValueError(error)
        # Flush pending reward-editor edits before training picks up paths.
        if getattr(self, "reward_controller", None) is not None:
            # Apply open rule edits into the in-memory spec.
            if self.reward_controller.selected_index is not None:
                # Persist editor widgets into the selected rule dict.
                self.reward_controller.apply_rule_edits()
        # Resolve pack so obs defaults stay aligned with the selected game.
        pack = load_game_pack(
            self.vars["game_id"].get(),
            games_root=self.vars["games_root"].get(),
        )
        # Build the dataclass used by the trainer.
        return TrainingConfig(
            game_id=self.vars["game_id"].get(),
            games_root=self.vars["games_root"].get(),
            rewards_path=self.vars["rewards_path"].get() or None,
            memory_path=self.vars["memory_path"].get() or None,
            rom_path=pack.rom_path,
            obs_width=pack.obs_width,
            obs_height=pack.obs_height,
            crop_top=pack.crop_top,
            algorithm=algorithm,
            n_envs=n_envs,
            total_timesteps=int(self.vars["total_timesteps"].get()),
            frame_skip=int(self.vars["frame_skip"].get()),
            learning_rate=float(self.vars["learning_rate"].get()),
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=int(self.vars["n_epochs"].get()),
            vec_env_type=self.vars["vec_env_type"].get(),
            visual_agents=int(self.vars["visual_agents"].get()),
            show_visuals=bool(self.vars["show_visuals"].get()),
            device=self.vars["device"].get(),
            resume_path=self.vars["resume_path"].get() or None,
            checkpoint_dir=self.vars["checkpoint_dir"].get(),
            log_dir=self.vars["log_dir"].get(),
            policy=default_policy_for_algorithm(algorithm),
            frame_stack=resolve_frame_stack(algorithm, None),
            pi_arch=self.vars["pi_arch"].get(),
            vf_arch=self.vars["vf_arch"].get(),
            q_arch=self.vars["q_arch"].get(),
            features_dim=int(self.vars["features_dim"].get()),
            lstm_hidden_size=int(self.vars["lstm_hidden_size"].get()),
            n_lstm_layers=int(self.vars["n_lstm_layers"].get()),
            shared_lstm=bool(self.vars["shared_lstm"].get()),
            enable_critic_lstm=bool(self.vars["enable_critic_lstm"].get()),
            activation=self.vars["activation"].get(),
            ortho_init=bool(self.vars["ortho_init"].get()),
        )

    def _set_training_controls(self, running):
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

    def _start_training(self):
        """Launch training on a background thread using the UI config."""
        # Ignore duplicate clicks while a run is already active.
        if self.is_training:
            # Warn the user that training is already running.
            messagebox.showinfo("Training", "Training is already running.")
            return
        # Materialize config from the form.
        try:
            # Validate buffer/batch geometry while building the config.
            config = self._build_config()
        except ValueError as exc:
            # Show a clear dialog when the GPU/batch settings are incompatible.
            messagebox.showerror("Invalid config", str(exc))
            # Abort the start request.
            return
        # Rebuild the preview grid if the visible agent count changed.
        self._rebuild_preview_grid(config.visual_agents)
        # Clear any previous stop request before launching.
        self.stop_event.clear()
        # Mark training as active.
        self.is_training = True
        # Flip Start/Stop button states.
        self._set_training_controls(True)
        # Update the status line.
        self.status_var.set(
            f"TRAINING {config.game_id} · {config.algorithm} · "
            f"{config.n_envs} ENVS · {len(config.resolve_state_paths())} STATES"
        )

        # Define the worker target that runs inside the thread.
        def _worker():
            # Best-effort training; surface failures in the UI.
            try:
                # Always pass the queue so telemetry arrives even without visuals.
                final_path = train_model(
                    config, frame_queue=self.frame_queue, stop_event=self.stop_event
                )
                # Detect whether the stop event caused an early exit.
                was_stopped = self.stop_event.is_set()
                # Schedule a completion update on the Tk thread.
                self.root.after(
                    0,
                    lambda: self._on_train_done(True, final_path, was_stopped),
                )
            except Exception as exc:
                # Schedule a failure update on the Tk thread.
                self.root.after(
                    0, lambda: self._on_train_done(False, str(exc), False)
                )

        # Start the background training thread.
        self.train_thread = threading.Thread(target=_worker, daemon=True)
        # Begin training.
        self.train_thread.start()

    def _stop_training(self):
        """Request a cooperative stop; callback returns False on next step."""
        # Ignore Stop clicks when nothing is running.
        if not self.is_training:
            # Nothing to stop.
            return
        # Signal the training callback to halt and save stopped_model.
        self.stop_event.set()
        # Update status so the user sees the request was received.
        self.status_var.set("STOPPING — WAITING FOR WORKERS TO EXIT CLEANLY...")
        # Log the stop request in the metrics pane.
        self.metrics.insert(tk.END, "> STOP requested — saving stopped_model.zip\n")
        # Disable Stop to prevent repeated clicks while draining.
        self.stop_btn.config(state=tk.DISABLED)

    def _on_train_done(self, success, detail, was_stopped):
        """Update the UI when the background training thread finishes."""
        # Clear the running flag.
        self.is_training = False
        # Restore Start / disable Stop.
        self._set_training_controls(False)
        # Clear the stop event for the next run.
        self.stop_event.clear()
        # Branch on success vs failure vs user stop.
        if success and was_stopped:
            # Show that training was halted by the user.
            self.status_var.set(f"STOPPED — SAVED {detail}")
            # Append a stop line to the metrics log.
            self.metrics.insert(tk.END, f"> Stopped cleanly → {detail}\n")
        elif success:
            # Show the final model path.
            self.status_var.set(f"COMPLETE — {detail}")
            # Append a success line to the metrics log.
            self.metrics.insert(tk.END, f"> Saved final model → {detail}\n")
        else:
            # Show the error message.
            self.status_var.set("ERROR — SEE LOG")
            # Append the exception text.
            self.metrics.insert(tk.END, f"> ERROR: {detail}\n")
            # Pop a dialog so the failure is obvious.
            messagebox.showerror("Training failed", detail)

    def _poll_frames(self):
        """Drain the UI queue for frames, stats dumps, and event tickers."""
        # Drain all currently queued payloads without blocking.
        try:
            # Loop until the queue is empty.
            while True:
                # Non-blocking get of the next payload.
                payload = self.frame_queue.get_nowait()
                # Route by payload type (stats / event / frames).
                self._handle_queue_payload(payload)
        except queue.Empty:
            # Nothing left to draw this tick.
            pass
        except Exception:
            # Ignore transient multiprocessing / Tk image errors.
            pass
        # Schedule the next poll.
        self.root.after(100, self._poll_frames)

    def _handle_queue_payload(self, payload):
        """Dispatch one queue message to stats, events, or live screens."""
        # Ignore malformed payloads.
        if not isinstance(payload, dict):
            # Nothing actionable.
            return
        # Read the multiplex discriminator (default keeps old frame payloads working).
        payload_type = payload.get("type", "frames")
        # SB3 logger.dump table → telemetry HUD.
        if payload_type == "stats":
            # Refresh ROLL / TIME / TRAIN cells.
            self.stats_board.update_stats(payload.get("stats") or {})
            # Also append a compact line to the text log.
            steps = payload.get("step", "?")
            # Keep the scrolling log short and useful.
            self.metrics.insert(tk.END, f"> stats dump @ step {steps}\n")
            # Autoscroll to the latest line.
            self.metrics.see(tk.END)
            # Done with this payload.
            return
        # Checkpoint / best-progress ticker line.
        if payload_type == "event":
            # Message text from the training callback.
            message = payload.get("message", "")
            # Update the olive ticker.
            self.stats_board.set_event(message)
            # Mirror into the scrolling log.
            self.metrics.insert(tk.END, f"> {message}\n")
            # Autoscroll to the latest line.
            self.metrics.see(tk.END)
            # Done with this payload.
            return
        # Live agent RGB frames (only when the user enabled visuals).
        if payload_type == "frames" and self.vars["show_visuals"].get():
            # Draw the agent screen grid.
            self._update_previews(payload)

    def _update_previews(self, payload):
        """Draw frames and captions for the visible agent panes."""
        # Delegate drawing to the shared preview helper.
        update_previews(
            self.preview_labels,
            self.caption_labels,
            self.photo_refs,
            payload,
            self.vars["visual_agents"].get(),
        )


def main():
    """Entry point for the multi-agent training dashboard."""
    mp.freeze_support()
    root = tk.Tk()
    TrainingDashboard(root)
    root.mainloop()


if __name__ == "__main__":
    main()
