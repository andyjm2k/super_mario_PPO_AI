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


def _load_photo(filename, width, height):
    """Load a cabinet image and return a Tk photo, or None."""
    # Absolute path next to this module.
    path = os.path.join(_ASSET_DIR, filename)
    # Missing art should not stop the UI.
    if not os.path.isfile(path):
        # Caller falls back to a text banner.
        return None
    # Open with Pillow.
    image = Image.open(path).convert("RGB")
    # Fit the slot.
    image = image.resize((int(width), int(height)), Image.LANCZOS)
    # Tk photo.
    return ImageTk.PhotoImage(image)


def build_marquee(parent, title, subtitle):
    """Paint the graffiti marquee and return the frame (keeps the photos)."""
    # Cabinet strip.
    frame = tk.Frame(parent, bg=GB_DARKEST, highlightthickness=0)
    # Wide marquee art.
    marquee = _load_photo("arcade_marquee.jpg", 1040, 168)
    # Side sticker.
    sticker = _load_photo("arcade_sticker.jpg", 120, 168)
    # Remember photos so Tk does not drop them.
    frame._photos = [photo for photo in (marquee, sticker) if photo is not None]
    # Art row.
    row = tk.Frame(frame, bg=GB_DARKEST)
    # Fill the banner width.
    row.pack(fill=tk.X)
    # Sticker on the left when it loaded.
    if sticker is not None:
        # Left graffiti panel.
        tk.Label(row, image=sticker, bg=GB_DARKEST, bd=0).pack(side=tk.LEFT, padx=(0, 6))
    # Marquee in the middle.
    if marquee is not None:
        # Host the painting and the title overlay.
        host = tk.Label(row, image=marquee, bg=GB_DARKEST, bd=0)
        # Let the title sit on the painting.
        host.pack(side=tk.LEFT, fill=tk.X, expand=True)
        # Magenta drop shadow.
        tk.Label(
            host,
            text=title,
            bg="#1a0a22",
            fg=GB_DARK,
            font=("Impact", 32),
            padx=8,
        ).place(relx=0.5, rely=0.38, anchor="center", x=3, y=3)
        # Yellow title on top of the shadow.
        tk.Label(
            host,
            text=title,
            bg="#1a0a22",
            fg=GB_LIGHTEST,
            font=("Impact", 32),
            padx=8,
        ).place(relx=0.5, rely=0.38, anchor="center")
        # Cyan tagline.
        tk.Label(
            host,
            text=subtitle,
            bg="#1a0a22",
            fg=GB_LIGHT,
            font=("Courier New", 10, "bold"),
            padx=6,
        ).place(relx=0.5, rely=0.72, anchor="center")
    else:
        # Text fallback when the jpg is missing.
        tk.Label(row, text=title, style="Title.TLabel").pack()
        # Subtitle under the fallback title.
        tk.Label(row, text=subtitle, style="Sub.TLabel").pack()
    # The frame the caller packs.
    return frame
