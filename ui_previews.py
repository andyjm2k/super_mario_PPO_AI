"""Live agent screen grid helpers for the training booth."""

import tkinter as tk
from tkinter import ttk

import numpy as np
from PIL import Image, ImageTk

from gb_theme import GB_DARK, GB_DARKEST, GB_LIGHTEST

# Max preview width in the booth (fits GBA 240 and GBC 160).
_PREVIEW_MAX_W = 240
# Max preview height in the booth (fits GBA 160 and GBC 144).
_PREVIEW_MAX_H = 160


def preview_size(height: int, width: int):
    """Return (width, height) fitted inside the booth preview box."""
    # Guard against empty / malformed frames.
    safe_h = max(1, int(height))
    # Guard width similarly.
    safe_w = max(1, int(width))
    # Only downscale when a frame exceeds the booth box (never upscale).
    scale = min(1.0, _PREVIEW_MAX_W / safe_w, _PREVIEW_MAX_H / safe_h)
    # Integer pixel size for Tk.
    return max(1, int(round(safe_w * scale))), max(1, int(round(safe_h * scale)))


def rebuild_preview_grid(grid_frame, count):
    """Recreate agent preview labels; return (labels, captions, photo_refs)."""
    # Destroy any previous preview widgets.
    for child in grid_frame.winfo_children():
        # Remove the old pane.
        child.destroy()
    # Fresh list of label widgets used as image targets.
    preview_labels = []
    # Fresh list of caption labels under each pane.
    caption_labels = []
    # Compute a roughly square grid layout.
    cols = max(1, int(np.ceil(np.sqrt(count))))
    # Create one pane per visible agent.
    for index in range(count):
        # Cell frame holds image + caption with thick bezel feel.
        cell = tk.Frame(
            grid_frame,
            bg=GB_DARK,
            highlightbackground=GB_DARKEST,
            highlightthickness=3,
            bd=0,
            padx=4,
            pady=4,
        )
        # Place the cell in the grid.
        cell.grid(row=index // cols, column=index % cols, sticky="nsew", padx=4, pady=4)
        # Image label starts blank on cream screen glass.
        label = tk.Label(
            cell,
            text=f"P{index + 1}",
            bg=GB_LIGHTEST,
            fg=GB_DARK,
            font=("Courier New", 10, "bold"),
            width=18,
            height=8,
        )
        # Pack the image label.
        label.pack()
        # Caption shows progress / reward once training starts.
        caption = ttk.Label(cell, text="WAITING...", style="Caption.TLabel")
        # Pack the caption under the image.
        caption.pack()
        # Keep references for later updates.
        preview_labels.append(label)
        # Keep caption references too.
        caption_labels.append(caption)
    # Return widgets plus empty photo slots.
    return preview_labels, caption_labels, [None] * count


def update_previews(preview_labels, caption_labels, photo_refs, payload, visual_agents):
    """Draw frames and captions for the visible agent panes."""
    # Unpack frames and infos from the callback payload.
    frames = payload.get("frames") or []
    # Infos carry progress / reward for captions.
    infos = payload.get("infos") or []
    # Limit drawing to the number of visible panes.
    visible = min(len(preview_labels), len(frames), visual_agents)
    # Update each visible agent pane.
    for index in range(visible):
        # Fetch the RGB frame for this worker.
        frame = frames[index]
        # Skip workers that have not produced a frame yet.
        if frame is None:
            # Leave the pane unchanged.
            continue
        # Ensure we have a contiguous uint8 RGB array.
        array = np.asarray(frame, dtype=np.uint8)
        # Require a HxWx3 layout before drawing.
        if array.ndim != 3 or array.shape[2] < 3:
            # Skip malformed payloads rather than crashing the UI.
            continue
        # Convert to a PIL image for Tk.
        image = Image.fromarray(array[:, :, :3])
        # Fit GBA (240x160) and GBC (160x144) without forcing square GB size.
        preview_w, preview_h = preview_size(array.shape[0], array.shape[1])
        # Nearest-neighbor upscale keeps the pixel look.
        image = image.resize((preview_w, preview_h), Image.NEAREST)
        # Create a Tk-compatible photo image.
        photo = ImageTk.PhotoImage(image)
        # Keep a reference so Tk does not garbage-collect it.
        photo_refs[index] = photo
        # Push the image into the label.
        preview_labels[index].configure(
            image=photo, text="", width=preview_w, height=preview_h
        )
        # Build a caption from the latest info dict when present.
        info = (
            infos[index] if index < len(infos) and isinstance(infos[index], dict) else {}
        )
        # Format progress and reward for the caption.
        state_name = str(info.get("start_state", "?")).replace("\\", "/").split("/")[-1]
        # Compact monospace caption under the bezel.
        caption = (
            f"P{index + 1} {info.get('progress_pct', 0):.0f}% "
            f"R{info.get('episode_reward', 0):.0f} {state_name}"
        )
        # Update the caption label text.
        caption_labels[index].configure(text=caption)
