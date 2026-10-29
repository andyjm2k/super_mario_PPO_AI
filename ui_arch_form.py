"""Model-architecture form builder for the Mario training booth."""

from tkinter import ttk

from ui_tooltips import MODEL_TOOLTIPS, attach_tooltip, tip_pair


def build_arch_form(parent, vars_map):
    """Add architecture controls into parent; returns nothing (vars_map driven)."""

    # Helper to place a labeled control row with matching tooltips.
    def add_row(row, label, widget, tip_key=None):
        # Left label in Courier.
        label_widget = ttk.Label(parent, text=label, style="GB.TLabel")
        # Place the label in the grid.
        label_widget.grid(row=row, column=0, sticky="w", pady=2)
        # Right control widget.
        widget.grid(row=row, column=1, sticky="ew", pady=2)
        # Look up the explanation for this architecture field.
        tip_text = MODEL_TOOLTIPS.get(tip_key or label)
        # Attach hover help when available.
        if tip_text:
            # Bind the same tip to label and control.
            tip_pair(label_widget, widget, tip_text)
        # Return the label for optional further use.
        return label_widget

    # Start at the first grid row.
    row = 0
    # Actor MLP sizes (comma-separated).
    add_row(
        row,
        "PI LAYERS",
        ttk.Entry(parent, textvariable=vars_map["pi_arch"], width=20, style="GB.TEntry"),
    )
    row += 1
    # Critic MLP sizes (comma-separated).
    add_row(
        row,
        "VF LAYERS",
        ttk.Entry(parent, textvariable=vars_map["vf_arch"], width=20, style="GB.TEntry"),
    )
    row += 1
    # Q-network sizes for QRDQN.
    add_row(
        row,
        "Q LAYERS",
        ttk.Entry(parent, textvariable=vars_map["q_arch"], width=20, style="GB.TEntry"),
    )
    row += 1
    # CNN feature dimension.
    add_row(
        row,
        "FEATURES DIM",
        ttk.Entry(
            parent, textvariable=vars_map["features_dim"], width=20, style="GB.TEntry"
        ),
    )
    row += 1
    # LSTM hidden size.
    add_row(
        row,
        "LSTM HIDDEN",
        ttk.Entry(
            parent,
            textvariable=vars_map["lstm_hidden_size"],
            width=20,
            style="GB.TEntry",
        ),
    )
    row += 1
    # Number of LSTM layers.
    add_row(
        row,
        "LSTM LAYERS",
        ttk.Spinbox(
            parent,
            from_=1,
            to=4,
            textvariable=vars_map["n_lstm_layers"],
            width=18,
            style="GB.TSpinbox",
        ),
    )
    row += 1
    # Activation function selector.
    add_row(
        row,
        "ACTIVATION",
        ttk.Combobox(
            parent,
            textvariable=vars_map["activation"],
            values=["relu", "tanh", "elu", "leaky_relu"],
            state="readonly",
            width=18,
            style="GB.TCombobox",
        ),
    )
    row += 1
    # Shared LSTM toggle for RecurrentPPO.
    shared_btn = ttk.Checkbutton(
        parent,
        text="SHARED LSTM",
        variable=vars_map["shared_lstm"],
        style="GB.TCheckbutton",
    )
    # Place the shared-LSTM toggle.
    shared_btn.grid(row=row, column=0, columnspan=2, sticky="w", pady=2)
    # Tooltip for shared LSTM.
    attach_tooltip(shared_btn, MODEL_TOOLTIPS["SHARED LSTM"])
    row += 1
    # Critic LSTM toggle for RecurrentPPO.
    critic_btn = ttk.Checkbutton(
        parent,
        text="CRITIC LSTM",
        variable=vars_map["enable_critic_lstm"],
        style="GB.TCheckbutton",
    )
    # Place the critic-LSTM toggle.
    critic_btn.grid(row=row, column=0, columnspan=2, sticky="w", pady=2)
    # Tooltip for critic LSTM.
    attach_tooltip(critic_btn, MODEL_TOOLTIPS["CRITIC LSTM"])
    row += 1
    # Orthogonal init toggle.
    ortho_btn = ttk.Checkbutton(
        parent,
        text="ORTHO INIT",
        variable=vars_map["ortho_init"],
        style="GB.TCheckbutton",
    )
    # Place the ortho-init toggle.
    ortho_btn.grid(row=row, column=0, columnspan=2, sticky="w", pady=2)
    # Tooltip for orthogonal init.
    attach_tooltip(ortho_btn, MODEL_TOOLTIPS["ORTHO INIT"])
    row += 1
    # Short help text under the architecture controls.
    help_label = ttk.Label(
        parent,
        text="Hover any label for details\nPI/VF/Q = comma sizes e.g. 512,256",
        style="Caption.TLabel",
        wraplength=240,
    )
    # Place the help caption.
    help_label.grid(row=row, column=0, columnspan=2, sticky="w", pady=(8, 0))
    # Stretch the value column.
    parent.columnconfigure(1, weight=1)
