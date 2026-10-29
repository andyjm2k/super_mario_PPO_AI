"""Game Boy Color ttk theme helpers for the training booth UI."""

from tkinter import ttk


# Classic DMG / olive Game Boy palette for the booth chrome.
GB_LIGHTEST = "#c4cfa1"
# Mid-light olive fill for panels.
GB_LIGHT = "#8b956d"
# Dark olive for borders and secondary text.
GB_DARK = "#4a5138"
# Near-black green for captions and chrome outlines.
GB_DARKEST = "#1e2410"
# Soft cream used behind the live screen bezels.
GB_CREAM = "#dbe4c0"
# Accent red for the STOP button (like a GBC face button).
GB_STOP = "#8b2e2e"
# Accent green for the START button.
GB_START = "#2e5a2e"


def apply_gb_theme(root):
    """Configure ttk styles for a chunky olive Game Boy look."""
    # Create a style object bound to this root.
    style = ttk.Style(root)
    # Start from clam so background colors actually paint on Windows.
    style.theme_use("clam")
    # Root frame background.
    style.configure("GB.TFrame", background=GB_DARKEST)
    # Cream panel background for nested sections.
    style.configure("Panel.TFrame", background=GB_CREAM)
    # LabelFrame chrome with thick olive borders.
    style.configure(
        "GB.TLabelframe",
        background=GB_CREAM,
        bordercolor=GB_DARK,
        borderwidth=4,
        relief="solid",
        labeloutside=False,
    )
    # LabelFrame title text in monospace dark olive.
    style.configure(
        "GB.TLabelframe.Label",
        background=GB_CREAM,
        foreground=GB_DARKEST,
        font=("Courier New", 11, "bold"),
    )
    # Standard labels inside panels.
    style.configure(
        "GB.TLabel",
        background=GB_CREAM,
        foreground=GB_DARKEST,
        font=("Courier New", 9),
    )
    # Status / caption labels under agent panes.
    style.configure(
        "Caption.TLabel",
        background=GB_CREAM,
        foreground=GB_DARK,
        font=("Courier New", 8),
    )
    # Title banner label.
    style.configure(
        "Title.TLabel",
        background=GB_DARKEST,
        foreground=GB_LIGHTEST,
        font=("Courier New", 16, "bold"),
    )
    # Subtitle under the banner.
    style.configure(
        "Sub.TLabel",
        background=GB_DARKEST,
        foreground=GB_LIGHT,
        font=("Courier New", 9),
    )
    # Entry fields with cream fill and dark text.
    style.configure(
        "GB.TEntry",
        fieldbackground=GB_LIGHTEST,
        foreground=GB_DARKEST,
        insertcolor=GB_DARKEST,
        bordercolor=GB_DARK,
        lightcolor=GB_LIGHT,
        darkcolor=GB_DARK,
        font=("Courier New", 9),
    )
    # Combobox matching the entry look.
    style.configure(
        "GB.TCombobox",
        fieldbackground=GB_LIGHTEST,
        foreground=GB_DARKEST,
        background=GB_LIGHT,
        arrowcolor=GB_DARKEST,
        font=("Courier New", 9),
    )
    # Spinbox matching the entry look.
    style.configure(
        "GB.TSpinbox",
        fieldbackground=GB_LIGHTEST,
        foreground=GB_DARKEST,
        background=GB_LIGHT,
        arrowcolor=GB_DARKEST,
        font=("Courier New", 9),
    )
    # Checkbutton with cream panel background.
    style.configure(
        "GB.TCheckbutton",
        background=GB_CREAM,
        foreground=GB_DARKEST,
        font=("Courier New", 9),
    )
    # Primary START button (green face-button vibe).
    style.configure(
        "Start.TButton",
        background=GB_START,
        foreground=GB_LIGHTEST,
        font=("Courier New", 11, "bold"),
        borderwidth=4,
        relief="raised",
        padding=6,
    )
    # Map pressed/disabled states for START.
    style.map(
        "Start.TButton",
        background=[("pressed", GB_DARK), ("disabled", GB_LIGHT)],
        foreground=[("disabled", GB_DARK)],
    )
    # STOP button (red face-button vibe).
    style.configure(
        "Stop.TButton",
        background=GB_STOP,
        foreground=GB_LIGHTEST,
        font=("Courier New", 11, "bold"),
        borderwidth=4,
        relief="raised",
        padding=6,
    )
    # Map pressed/disabled states for STOP.
    style.map(
        "Stop.TButton",
        background=[("pressed", GB_DARK), ("disabled", GB_LIGHT)],
        foreground=[("disabled", GB_DARK)],
    )
    # Neutral browse button.
    style.configure(
        "GB.TButton",
        background=GB_LIGHT,
        foreground=GB_DARKEST,
        font=("Courier New", 9, "bold"),
        borderwidth=3,
        relief="raised",
        padding=4,
    )
    # Notebook chrome for TRAIN / MODEL tabs.
    style.configure(
        "TNotebook",
        background=GB_DARKEST,
        borderwidth=0,
    )
    # Individual notebook tabs.
    style.configure(
        "TNotebook.Tab",
        background=GB_LIGHT,
        foreground=GB_DARKEST,
        font=("Courier New", 9, "bold"),
        padding=[8, 4],
    )
    # Highlight the selected tab.
    style.map(
        "TNotebook.Tab",
        background=[("selected", GB_CREAM)],
        foreground=[("selected", GB_DARKEST)],
    )
    # Return the configured style for optional further tweaks.
    return style
