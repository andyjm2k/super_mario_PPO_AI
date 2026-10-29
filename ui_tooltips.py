"""Hover tooltips for the Mario training booth controls."""

import tkinter as tk

from gb_theme import GB_CREAM, GB_DARKEST, GB_LIGHTEST


# Explanations for TRAIN-tab settings.
TRAIN_TOOLTIPS = {
    "GAME": (
        "Which game pack to train.\n"
        "Each pack under games/<id>/ has its own ROM, states/, "
        "memory.json, and rewards.json so titles never co-mingle."
    ),
    "ALGORITHM": (
        "Which RL algorithm trains the agent.\n"
        "• RecurrentPPO — CNN + LSTM; best for multi-level memory\n"
        "• PPO — feed-forward CNN; faster, less memory across stages\n"
        "• QRDQN — off-policy value learning; different update style"
    ),
    "PARALLEL ENVS": (
        "How many PyBoy games run at once.\n"
        "More envs = more experience per second and larger rollout "
        "batches for the GPU, but needs more CPU RAM."
    ),
    "VEC ENV": (
        "How parallel envs are launched.\n"
        "• subproc — true multiprocessing (recommended)\n"
        "• dummy — single-process; slower, easier to debug"
    ),
    "TIMESTEPS": (
        "Total environment steps to train for before stopping.\n"
        "Larger values train longer. Example: 2,000,000 ≈ a long run."
    ),
    "FRAME SKIP": (
        "How many emulator frames each agent action is held.\n"
        "Higher skip = faster wall-clock training but coarser control.\n"
        "Typical Mario values: 2–4."
    ),
    "LEARN RATE": (
        "Optimizer step size for updating network weights.\n"
        "Too high can make training unstable; too low learns slowly.\n"
        "Default ~0.00025 works well for PPO-family algos."
    ),
    "N_STEPS": (
        "Rollout length per env before each weight update.\n"
        "Buffer size = PARALLEL ENVS × N_STEPS.\n"
        "Longer rollouts = less frequent updates, more data each time."
    ),
    "BATCH": (
        "Minibatch size during the weight-update (train) phase.\n"
        "Must divide (PARALLEL ENVS × N_STEPS).\n"
        "Larger batches usually raise GPU utilization."
    ),
    "N_EPOCHS": (
        "How many times each rollout buffer is reused for updates.\n"
        "More epochs = more GPU work per collection cycle.\n"
        "Too many can overfit that rollout."
    ),
    "DEVICE": (
        "Where policy updates run.\n"
        "• auto — CUDA if available, else CPU\n"
        "• cuda — force GPU\n"
        "• cpu — force CPU (slow for CNNs)"
    ),
    "VIS AGENTS": (
        "How many live agent screens to show in the booth.\n"
        "Only used when SHOW LIVE SCREENS is on.\n"
        "Fewer panes leave more CPU for training workers."
    ),
    "SHOW LIVE SCREENS": (
        "Stream live RGB frames from workers into the UI.\n"
        "Useful for watching agents, but costs CPU.\n"
        "Turn off for maximum training throughput."
    ),
    "RESUME": (
        "Optional path to a previous .zip checkpoint.\n"
        "Training continues from those weights instead of random init.\n"
        "Leave blank to train from scratch."
    ),
    "BROWSE .ZIP": (
        "Open a file dialog to pick a checkpoint zip to resume from."
    ),
    "GPU PRESET (5090+)": (
        "One-click load of high-throughput settings for large GPUs:\n"
        "more envs, larger batch/epochs, CUDA, live screens off."
    ),
    "▶ START": (
        "Start training with the current TRAIN + MODEL settings."
    ),
    "■ STOP": (
        "Request a clean stop. Saves stopped_model and shuts down workers."
    ),
}


# Explanations for MODEL-tab architecture settings.
MODEL_TOOLTIPS = {
    "PI LAYERS": (
        "Actor (policy) MLP hidden sizes after the CNN/LSTM.\n"
        "Comma-separated, e.g. 512,256 = two layers with 512 then 256 units.\n"
        "Wider/deeper = more capacity, more GPU work, more overfitting risk."
    ),
    "VF LAYERS": (
        "Critic (value) MLP hidden sizes after the CNN/LSTM.\n"
        "Comma-separated, e.g. 512,512.\n"
        "Estimates expected return; often matched to PI LAYERS."
    ),
    "Q LAYERS": (
        "Q-network MLP sizes for QRDQN only.\n"
        "Comma-separated, e.g. 512,256.\n"
        "Ignored by PPO / RecurrentPPO."
    ),
    "FEATURES DIM": (
        "Output size of the NatureCNN feature extractor.\n"
        "Larger = richer visual features and heavier GPU updates.\n"
        "Typical range: 256–1024."
    ),
    "LSTM HIDDEN": (
        "Hidden/cell width of the RecurrentPPO LSTM.\n"
        "Larger memory helps across different level layouts.\n"
        "Only used by RecurrentPPO."
    ),
    "LSTM LAYERS": (
        "How many LSTM layers are stacked.\n"
        "1 is standard; 2+ is heavier and often unnecessary.\n"
        "Only used by RecurrentPPO."
    ),
    "ACTIVATION": (
        "Non-linearity between MLP layers.\n"
        "• relu — default, fast\n"
        "• tanh — smoother, sometimes more stable\n"
        "• elu / leaky_relu — alternatives for vanishing gradients"
    ),
    "SHARED LSTM": (
        "If on, actor and critic share one LSTM (RecurrentPPO).\n"
        "Saves compute; off lets each head keep its own memory."
    ),
    "CRITIC LSTM": (
        "If on, the critic also uses an LSTM (RecurrentPPO).\n"
        "Usually leave on so value estimates see temporal context."
    ),
    "ORTHO INIT": (
        "Orthogonal weight initialization for policy networks.\n"
        "Often improves PPO training stability. Leave on unless debugging."
    ),
}


class HoverTooltip:
    """Show a delayed olive tooltip near the cursor on hover."""

    def __init__(self, widget, text, delay_ms=450):
        # Widget that owns this tooltip.
        self.widget = widget
        # Help text shown in the floating window.
        self.text = text
        # Delay before the tip appears (avoids flicker while moving).
        self.delay_ms = delay_ms
        # Scheduled after() id while waiting to show.
        self._after_id = None
        # Floating Toplevel while the tip is visible.
        self._tip_window = None
        # Show tip after hover delay.
        self.widget.bind("<Enter>", self._schedule, add="+")
        # Hide tip when the pointer leaves.
        self.widget.bind("<Leave>", self._hide, add="+")
        # Hide tip when the widget is destroyed.
        self.widget.bind("<Destroy>", self._hide, add="+")
        # Hide tip on click so it does not block interaction.
        self.widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _event=None):
        """Start the delay timer when the pointer enters."""
        # Cancel any previous pending show.
        self._cancel_timer()
        # Schedule the tip to appear after delay_ms.
        self._after_id = self.widget.after(self.delay_ms, self._show)

    def _cancel_timer(self):
        """Cancel a pending delayed show if one exists."""
        # Only cancel when a timer is outstanding.
        if self._after_id is not None:
            # Drop the scheduled callback.
            self.widget.after_cancel(self._after_id)
            # Clear the timer id.
            self._after_id = None

    def _show(self):
        """Create and place the tooltip window."""
        # Do nothing if a tip is already open.
        if self._tip_window is not None:
            # Already visible.
            return
        # Clear the timer id now that we are showing.
        self._after_id = None
        # Anchor near the bottom-right of the widget.
        x = self.widget.winfo_rootx() + 20
        # Place slightly below the widget.
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        # Create a borderless floating window.
        tip = tk.Toplevel(self.widget)
        # Keep it above the booth without stealing focus.
        tip.wm_overrideredirect(True)
        # Position the tip on screen.
        tip.wm_geometry(f"+{x}+{y}")
        # Try to keep the tip above siblings on Windows.
        try:
            # Topmost helps the tip stay readable over the booth.
            tip.attributes("-topmost", True)
        except tk.TclError:
            # Some platforms reject -topmost; ignore safely.
            pass
        # Cream label with dark text matching the GB theme.
        label = tk.Label(
            tip,
            text=self.text,
            justify=tk.LEFT,
            background=GB_CREAM,
            foreground=GB_DARKEST,
            relief=tk.SOLID,
            borderwidth=2,
            font=("Courier New", 8),
            padx=8,
            pady=6,
            wraplength=280,
        )
        # Pack the label to fill the tip window.
        label.pack()
        # Remember the tip so we can destroy it later.
        self._tip_window = tip

    def _hide(self, _event=None):
        """Cancel timers and destroy the tip window."""
        # Stop any pending delayed show.
        self._cancel_timer()
        # Destroy the floating window when present.
        if self._tip_window is not None:
            # Close the tip UI.
            self._tip_window.destroy()
            # Clear the reference.
            self._tip_window = None


def attach_tooltip(widget, text):
    """Attach a HoverTooltip to one widget and return the tooltip object."""
    # Build and return the tooltip so callers can keep a reference if needed.
    return HoverTooltip(widget, text)


def tip_pair(label_widget, control_widget, text):
    """Attach the same tooltip text to a label and its control."""
    # Tip on the text label (users hover the name).
    attach_tooltip(label_widget, text)
    # Tip on the entry/combo/spinbox itself.
    attach_tooltip(control_widget, text)
