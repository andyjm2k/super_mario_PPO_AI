"""Arcade-cabinet ttk theme and marquee art for the trainer UI."""

import os
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk


# Cabinet black behind the whole window.
GB_DARKEST = "#120814"
# Deep purple panels behind forms.
GB_CREAM = "#2a1844"
# Neon magenta outlines.
GB_DARK = "#ff2bd6"
# Cyan bulbs and secondary type.
GB_LIGHT = "#3dfff3"
# Yellow ink for titles and form text.
GB_LIGHTEST = "#ffe14a"
# CRT glass behind logs and score cells.
GB_FIELD = "#0e0718"
# Hot pink STOP coin.
GB_STOP = "#ff2d6a"
# Mint START coin.
GB_START = "#14f195"
# Folder of generated cabinet art, next to this file.
_ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def apply_gb_theme(root):
    """Paint ttk widgets as a neon arcade cabinet."""
    # Style object for this window.
    style = ttk.Style(root)
    # Clam lets us set real background colors on Windows.
    style.theme_use("clam")
    # Chassis frames.
    style.configure("GB.TFrame", background=GB_DARKEST)
    # Form panels.
    style.configure("Panel.TFrame", background=GB_CREAM)
    # Thick neon labelframe.
    style.configure(
        "GB.TLabelframe",
        background=GB_CREAM,
        bordercolor=GB_DARK,
        lightcolor=GB_LIGHT,
        darkcolor=GB_DARK,
        borderwidth=4,
        relief="solid",
    )
    # Labelframe titles in cyan.
    style.configure(
        "GB.TLabelframe.Label",
        background=GB_CREAM,
        foreground=GB_LIGHT,
        font=("Impact", 12),
    )
    # Form labels in yellow.
    style.configure(
        "GB.TLabel",
        background=GB_CREAM,
        foreground=GB_LIGHTEST,
        font=("Courier New", 9, "bold"),
    )
    # Captions under the screens.
    style.configure(
        "Caption.TLabel",
        background=GB_CREAM,
        foreground=GB_LIGHT,
        font=("Courier New", 8, "bold"),
    )
    # Captions painted on the black graffiti bezel.
    style.configure(
        "Booth.TLabel",
        background=GB_DARKEST,
        foreground=GB_LIGHT,
        font=("Courier New", 8, "bold"),
    )
    # Marquee title, used if the image fails to load.
    style.configure(
        "Title.TLabel",
        background=GB_DARKEST,
        foreground=GB_LIGHTEST,
        font=("Impact", 28),
    )
    # Subtitle under the marquee.
    style.configure(
        "Sub.TLabel",
        background=GB_DARKEST,
        foreground=GB_LIGHT,
        font=("Courier New", 10, "bold"),
    )
    # Dark fields with yellow type.
    style.configure(
        "GB.TEntry",
        fieldbackground=GB_FIELD,
        foreground=GB_LIGHTEST,
        insertcolor=GB_LIGHTEST,
        bordercolor=GB_DARK,
        lightcolor=GB_LIGHT,
        darkcolor=GB_DARK,
        font=("Courier New", 9),
    )
    # Matching combobox.
    style.configure(
        "GB.TCombobox",
        fieldbackground=GB_FIELD,
        foreground=GB_LIGHTEST,
        background=GB_CREAM,
        arrowcolor=GB_LIGHT,
        bordercolor=GB_DARK,
        font=("Courier New", 9),
    )
    # Matching spinbox.
    style.configure(
        "GB.TSpinbox",
        fieldbackground=GB_FIELD,
        foreground=GB_LIGHTEST,
        background=GB_CREAM,
        arrowcolor=GB_LIGHT,
        bordercolor=GB_DARK,
        font=("Courier New", 9),
    )
    # Checkbuttons on the purple panel.
    style.configure(
        "GB.TCheckbutton",
        background=GB_CREAM,
        foreground=GB_LIGHTEST,
        font=("Courier New", 9, "bold"),
    )
    # Mint insert-coin start button.
    style.configure(
        "Start.TButton",
        background=GB_START,
        foreground=GB_DARKEST,
        font=("Impact", 16),
        borderwidth=4,
        relief="raised",
        padding=6,
    )
    # Pressed and disabled start states.
    style.map(
        "Start.TButton",
        background=[("pressed", GB_LIGHT), ("disabled", "#355544")],
        foreground=[("disabled", GB_DARKEST)],
    )
    # Hot pink stop button.
    style.configure(
        "Stop.TButton",
        background=GB_STOP,
        foreground=GB_LIGHTEST,
        font=("Impact", 16),
        borderwidth=4,
        relief="raised",
        padding=6,
    )
    # Pressed and disabled stop states.
    style.map(
        "Stop.TButton",
        background=[("pressed", GB_DARK), ("disabled", "#422")],
        foreground=[("disabled", GB_LIGHTEST)],
    )
    # Small cabinet buttons.
    style.configure(
        "GB.TButton",
        background=GB_LIGHT,
        foreground=GB_DARKEST,
        font=("Impact", 12),
        borderwidth=3,
        relief="raised",
        padding=4,
    )
    # Notebook sits on the cabinet.
    style.configure("TNotebook", background=GB_DARKEST, borderwidth=0)
    # Unselected tabs.
    style.configure(
        "TNotebook.Tab",
        background="#3a1858",
        foreground=GB_LIGHT,
        font=("Impact", 12),
        padding=[12, 6],
    )
    # Selected tab is the magenta graffiti plaque.
    style.map(
        "TNotebook.Tab",
        background=[("selected", GB_DARK)],
        foreground=[("selected", GB_LIGHTEST)],
    )
    # Return the style object.
    return style


def fit_size(src_w, src_h, max_w, max_h):
    """Return a size that fits inside the box without stretching."""
    # Avoid a zero source.
    safe_w = max(1, int(src_w))
    # Avoid a zero source height.
    safe_h = max(1, int(src_h))
    # The smaller axis wins so neither side overflows.
    scale = min(float(max_w) / safe_w, float(max_h) / safe_h)
    # Integer pixels for Tk and OpenCV.
    return max(1, int(round(safe_w * scale))), max(1, int(round(safe_h * scale)))


def graffiti_bezel(parent):
    """Magenta, cyan, and yellow spray border. Returns (shell, inner)."""
    # Outer magenta coat.
    shell = tk.Frame(parent, bg=GB_DARK, bd=0, highlightthickness=0)
    # Black gap so the next ring reads as a separate spray line.
    gap = tk.Frame(shell, bg=GB_DARKEST, bd=0, highlightthickness=0)
    # Inset from the magenta edge.
    gap.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)
    # Cyan overspray.
    cyan = tk.Frame(gap, bg=GB_LIGHT, bd=0, highlightthickness=0)
    # Inset from the black gap.
    cyan.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
    # Thin yellow highlight, like a paint edge.
    yellow = tk.Frame(cyan, bg=GB_LIGHTEST, bd=0, highlightthickness=0)
    # Inset from the cyan edge.
    yellow.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
    # Cabinet black inside the bezel.
    inner = tk.Frame(yellow, bg=GB_DARKEST, bd=0, highlightthickness=0)
    # Inset from the yellow edge.
    inner.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
    # Keep the layers alive with the shell.
    shell._layers = (gap, cyan, yellow, inner)
    # Caller packs the shell and fills the inner frame.
    return shell, inner


def _load_photo(filename, max_width, max_height):
    """Load cabinet art fitted inside the box. Aspect ratio stays intact."""
    # Absolute path next to this module.
    path = os.path.join(_ASSET_DIR, filename)
    # Missing art should not stop the UI.
    if not os.path.isfile(path):
        # Caller falls back to a text banner.
        return None
    # Open with Pillow.
    image = Image.open(path).convert("RGB")
    # Uniform scale into the slot.
    fitted_w, fitted_h = fit_size(image.width, image.height, max_width, max_height)
    # Resize only when the slot differs from the file.
    if (fitted_w, fitted_h) != image.size:
        # High-quality downscale or mild upscale, never a stretch.
        image = image.resize((fitted_w, fitted_h), Image.LANCZOS)
    # Tk photo at the fitted size.
    return ImageTk.PhotoImage(image)


def build_marquee(parent, title, subtitle):
    """Paint the graffiti marquee and return the frame (keeps the photos)."""
    # Cabinet strip.
    frame = tk.Frame(parent, bg=GB_DARKEST, highlightthickness=0)
    # Graffiti bezel around the whole marquee.
    shell, inner = graffiti_bezel(frame)
    # The bezel spans the window.
    shell.pack(fill=tk.X)
    # The painting is 16:9. Cap the height so the header stays a header.
    marquee = _load_photo("arcade_marquee.jpg", 1040, 210)
    # Match the sticker to that height. Width follows its own ratio.
    sticker_h = marquee.height() if marquee is not None else 210
    # A wide cap that the portrait sticker will not reach.
    sticker = _load_photo("arcade_sticker.jpg", sticker_h, sticker_h)
    # Remember photos so Tk does not drop them.
    frame._photos = [photo for photo in (marquee, sticker) if photo is not None]
    # Art row, centered, so a wide window does not stretch the labels.
    row = tk.Frame(inner, bg=GB_DARKEST)
    # Anchor keeps the pictures at their real pixel size.
    row.pack(pady=(8, 4))
    # Sticker on the left when it loaded.
    if sticker is not None:
        # Left graffiti panel. No fill, so Tk cannot stretch it.
        tk.Label(row, image=sticker, bg=GB_DARKEST, bd=0).pack(side=tk.LEFT, padx=(8, 8))
    # Marquee beside it.
    if marquee is not None:
        # The photo is already the right aspect. Do not expand the label.
        tk.Label(row, image=marquee, bg=GB_DARKEST, bd=0).pack(side=tk.LEFT, padx=(0, 8))
    # Nameplate under the art, so the title is not a box painted over the picture.
    plate = tk.Frame(inner, bg=GB_DARKEST, height=36)
    # Fixed height so the shadow and the title share one line.
    plate.pack(fill=tk.X)
    # Stop the empty frame from collapsing.
    plate.pack_propagate(False)
    # Magenta offset, like a spray shadow.
    tk.Label(
        plate,
        text=title,
        bg=GB_DARKEST,
        fg=GB_DARK,
        font=("Impact", 28),
    ).place(relx=0.5, rely=0.5, anchor="center", x=2, y=2)
    # Yellow title on top of the shadow.
    tk.Label(
        plate,
        text=title,
        bg=GB_DARKEST,
        fg=GB_LIGHTEST,
        font=("Impact", 28),
    ).place(relx=0.5, rely=0.5, anchor="center")
    # Cyan tagline under the name.
    tk.Label(
        inner,
        text=subtitle,
        bg=GB_DARKEST,
        fg=GB_LIGHT,
        font=("Courier New", 10, "bold"),
    ).pack(pady=(0, 8))
    # The frame the caller packs.
    return frame
