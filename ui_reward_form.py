"""REWARDS tab: edit declarative reward rules + import memory maps."""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from pdt.memory_bank import MemoryBank
from pdt.reward_engine import RewardSpec
from ui_reward_schema import RULE_FIELDS, RULE_TYPES, default_rule_for_type


def build_reward_form(parent, controller):
    """Build the rewards editor UI; controller must expose pack reload helpers."""
    # Outer frame fills the notebook tab.
    frame = ttk.Frame(parent, style="Panel.TFrame")
    # Pack to fill the tab.
    frame.pack(fill=tk.BOTH, expand=True)
    # Top toolbar for import / save / reload.
    toolbar = ttk.Frame(frame, style="Panel.TFrame")
    # Pack toolbar across the top.
    toolbar.pack(fill=tk.X, pady=(0, 6))
    # Import memory JSON button.
    ttk.Button(
        toolbar,
        text="IMPORT MEMORY",
        style="GB.TButton",
        command=controller.import_memory_file,
    ).pack(side=tk.LEFT, padx=(0, 4))
    # Reload pack rewards from disk.
    ttk.Button(
        toolbar,
        text="RELOAD",
        style="GB.TButton",
        command=controller.reload_rewards,
    ).pack(side=tk.LEFT, padx=(0, 4))
    # Save rewards back to the game pack.
    ttk.Button(
        toolbar,
        text="SAVE REWARDS",
        style="GB.TButton",
        command=controller.save_rewards,
    ).pack(side=tk.LEFT, padx=(0, 4))
    # Memory path status label.
    controller.memory_status = tk.StringVar(value="MEMORY: pack default")
    # Show memory status.
    ttk.Label(toolbar, textvariable=controller.memory_status, style="Sub.TLabel").pack(
        side=tk.RIGHT
    )

    # Clip bounds row.
    clips = ttk.Frame(frame, style="Panel.TFrame")
    # Pack clip editors.
    clips.pack(fill=tk.X, pady=(0, 6))
    # Clip min label.
    ttk.Label(clips, text="CLIP MIN", style="GB.TLabel").pack(side=tk.LEFT)
    # Clip min entry.
    ttk.Entry(clips, textvariable=controller.clip_min_var, width=8).pack(
        side=tk.LEFT, padx=(4, 12)
    )
    # Clip max label.
    ttk.Label(clips, text="CLIP MAX", style="GB.TLabel").pack(side=tk.LEFT)
    # Clip max entry.
    ttk.Entry(clips, textvariable=controller.clip_max_var, width=8).pack(
        side=tk.LEFT, padx=4
    )

    # Body: rule list on the left, editor on the right.
    body = ttk.Frame(frame, style="Panel.TFrame")
    # Expand body.
    body.pack(fill=tk.BOTH, expand=True)
    # Left list column.
    left = ttk.Frame(body, style="Panel.TFrame")
    # Dock left.
    left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))
    # Rule listbox.
    controller.rule_list = tk.Listbox(
        left,
        height=16,
        width=22,
        bg="#0f380f",
        fg="#9bbc0f",
        selectbackground="#306230",
        font=("Courier New", 9),
    )
    # Pack listbox.
    controller.rule_list.pack(fill=tk.BOTH, expand=True)
    # Bind selection changes.
    controller.rule_list.bind("<<ListboxSelect>>", controller.on_rule_select)
    # List action buttons.
    list_btns = ttk.Frame(left, style="Panel.TFrame")
    # Pack under the list.
    list_btns.pack(fill=tk.X, pady=(4, 0))
    # Add rule.
    ttk.Button(
        list_btns, text="ADD", style="GB.TButton", command=controller.add_rule
    ).pack(side=tk.LEFT, padx=(0, 2))
    # Remove rule.
    ttk.Button(
        list_btns, text="DEL", style="GB.TButton", command=controller.remove_rule
    ).pack(side=tk.LEFT, padx=2)
    # New rule type selector.
    controller.new_rule_type = tk.StringVar(value=RULE_TYPES[0])
    # Combobox of rule types.
    ttk.Combobox(
        list_btns,
        textvariable=controller.new_rule_type,
        values=RULE_TYPES,
        width=16,
        state="readonly",
    ).pack(side=tk.LEFT, padx=(4, 0))

    # Right editor column.
    right = ttk.LabelFrame(body, text=" RULE EDITOR ", style="GB.TLabelframe", padding=8)
    # Expand editor.
    right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    # Scrollable editor host.
    controller.editor_host = ttk.Frame(right, style="Panel.TFrame")
    # Pack editor host.
    controller.editor_host.pack(fill=tk.BOTH, expand=True)
    # Apply-edits button.
    ttk.Button(
        right,
        text="APPLY RULE EDITS",
        style="GB.TButton",
        command=controller.apply_rule_edits,
    ).pack(anchor="e", pady=(8, 0))
    # Return the root frame.
    return frame


class RewardFormController:
    """State + handlers for the declarative rewards editor."""

    def __init__(self, root, vars_map, on_status=None):
        # Tk root for dialogs.
        self.root = root
        # Shared training UI variables (includes game_id).
        self.vars = vars_map
        # Optional status callback into the main metrics log.
        self.on_status = on_status or (lambda _msg: None)
        # Loaded reward spec (mutable).
        self.reward_spec: RewardSpec | None = None
        # Loaded / imported memory bank.
        self.memory_bank: MemoryBank | None = None
        # Signal names for dropdowns.
        self.signal_names = []
        # Currently selected rule index.
        self.selected_index = None
        # Dynamic editor StringVars keyed by field name.
        self.field_vars = {}
        # Clip vars created before build.
        self.clip_min_var = tk.StringVar(value="-50")
        # Clip max.
        self.clip_max_var = tk.StringVar(value="50")
        # Widgets filled by build_reward_form.
        self.rule_list = None
        # Editor host frame.
        self.editor_host = None
        # Memory status var.
        self.memory_status = tk.StringVar(value="MEMORY: pack default")
        # New rule type var.
        self.new_rule_type = tk.StringVar(value=RULE_TYPES[0])
        # Enabled checkbox var.
        self.enabled_var = tk.BooleanVar(value=True)
        # Rule id entry.
        self.rule_id_var = tk.StringVar(value="")

    def load_from_pack(self, pack):
        """Load rewards + memory from a resolved GamePack."""
        # Load declarative rewards.
        self.reward_spec = pack.load_rewards()
        # Load memory addresses.
        self.memory_bank = pack.load_memory()
        # Refresh signal dropdown values.
        self.signal_names = self.memory_bank.signal_names()
        # Sync clip editors.
        self.clip_min_var.set(str(self.reward_spec.data.get("clip_min", -50)))
        # Clip max.
        self.clip_max_var.set(str(self.reward_spec.data.get("clip_max", 50)))
        # Remember paths on the shared vars map for training.
        self.vars["rewards_path"].set(self.reward_spec.source_path or "")
        # Memory path for training overrides.
        self.vars["memory_path"].set(self.memory_bank.source_path or "")
        # Update status strip.
        self.memory_status.set(f"MEMORY: {pack.game_id}")
        # Refresh the listbox.
        self.refresh_rule_list()
        # Clear editor.
        self.clear_editor()
        # Log.
        self.on_status(f"Loaded rewards for {pack.name}")

    def refresh_rule_list(self):
        """Rebuild the rule listbox from the current spec."""
        # Nothing to do before widgets exist.
        if self.rule_list is None or self.reward_spec is None:
            # Bail out.
            return
        # Clear existing entries.
        self.rule_list.delete(0, tk.END)
        # Insert each rule summary.
        for rule in self.reward_spec.rules:
            # Enabled marker.
            mark = "✓" if rule.get("enabled", True) else "·"
            # Rule id + type.
            label = f"{mark} {rule.get('id', '?')} [{rule.get('type', '?')}]"
            # Append to the listbox.
            self.rule_list.insert(tk.END, label)

    def on_rule_select(self, _event=None):
        """Populate the editor when the user selects a rule."""
        # Require a list selection.
        if not self.rule_list.curselection():
            # Nothing selected.
            return
        # Selected index.
        index = int(self.rule_list.curselection()[0])
        # Remember selection.
        self.selected_index = index
        # Fetch the rule dict.
        rule = self.reward_spec.rules[index]
        # Rebuild editor widgets for this type.
        self.build_editor_for_rule(rule)

    def clear_editor(self):
        """Remove all widgets from the editor host."""
        # Skip when host is missing.
        if self.editor_host is None:
            # Nothing to clear.
            return
        # Destroy children.
        for child in self.editor_host.winfo_children():
            # Remove widget.
            child.destroy()
        # Clear field var map.
        self.field_vars = {}

    def build_editor_for_rule(self, rule):
        """Create labeled editors for the selected rule type."""
        # Clear previous fields.
        self.clear_editor()
        # Rule id.
        self.rule_id_var.set(str(rule.get("id", "")))
        # Enabled flag.
        self.enabled_var.set(bool(rule.get("enabled", True)))
        # Row counter.
        row = 0
        # ID label.
        ttk.Label(self.editor_host, text="ID", style="GB.TLabel").grid(
            row=row, column=0, sticky="w", pady=2
        )
        # ID entry.
        ttk.Entry(self.editor_host, textvariable=self.rule_id_var, width=22).grid(
            row=row, column=1, sticky="ew", pady=2
        )
        row += 1
        # Enabled checkbox.
        ttk.Checkbutton(
            self.editor_host,
            text="ENABLED",
            variable=self.enabled_var,
            style="GB.TCheckbutton",
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=2)
        row += 1
        # Type label (read-only).
        ttk.Label(self.editor_host, text="TYPE", style="GB.TLabel").grid(
            row=row, column=0, sticky="w", pady=2
        )
        # Type value.
        ttk.Label(
            self.editor_host, text=str(rule.get("type", "")), style="Sub.TLabel"
        ).grid(row=row, column=1, sticky="w", pady=2)
        row += 1
        # Field definitions for this type.
        fields = RULE_FIELDS.get(rule.get("type"), [])
        # Build each field editor.
        for label, key, kind in fields:
            # Field label.
            ttk.Label(self.editor_host, text=label, style="GB.TLabel").grid(
                row=row, column=0, sticky="w", pady=2
            )
            # Current value as string.
            current = rule.get(key, "")
            # Bool fields use a BooleanVar.
            if kind == "bool":
                # Boolean variable.
                var = tk.BooleanVar(value=bool(current))
                # Checkbox control.
                ttk.Checkbutton(
                    self.editor_host, variable=var, style="GB.TCheckbutton"
                ).grid(row=row, column=1, sticky="w", pady=2)
            elif kind == "signal":
                # String var for signal name.
                var = tk.StringVar(value=str(current or ""))
                # Combobox of known memory signals (+ blank).
                values = [""] + list(self.signal_names)
                # Signal picker.
                ttk.Combobox(
                    self.editor_host,
                    textvariable=var,
                    values=values,
                    width=20,
                ).grid(row=row, column=1, sticky="ew", pady=2)
            else:
                # Numeric / text entry.
                var = tk.StringVar(value="" if current is None else str(current))
                # Entry widget.
                ttk.Entry(self.editor_host, textvariable=var, width=22).grid(
                    row=row, column=1, sticky="ew", pady=2
                )
            # Remember var + kind for apply.
            self.field_vars[key] = (var, kind)
            # Next row.
            row += 1
        # Stretch editor columns.
        self.editor_host.columnconfigure(1, weight=1)

    def apply_rule_edits(self):
        """Write editor fields back into the selected rule dict."""
        # Need a selection + spec.
        if self.selected_index is None or self.reward_spec is None:
            # Nothing to apply.
            return
        # Target rule.
        rule = self.reward_spec.rules[self.selected_index]
        # Update id / enabled.
        rule["id"] = self.rule_id_var.get().strip() or rule.get("id", "rule")
        # Enabled flag.
        rule["enabled"] = bool(self.enabled_var.get())
        # Apply each dynamic field.
        for key, (var, kind) in self.field_vars.items():
            # Raw string / bool from the widget.
            raw = var.get()
            # Bool fields store booleans.
            if kind == "bool":
                # Cast to bool.
                rule[key] = bool(raw)
            elif kind == "int":
                # Parse integer (blank -> remove optional keys for also_signal etc.).
                text = str(raw).strip()
                # Allow blank optional ints to become None / delete.
                if text == "":
                    # Drop empty optional.
                    rule.pop(key, None)
                else:
                    # Store int.
                    rule[key] = int(float(text))
            elif kind == "float":
                # Parse float.
                text = str(raw).strip()
                # Blank optional float.
                if text == "":
                    # Drop empty.
                    rule.pop(key, None)
                else:
                    # Store float.
                    rule[key] = float(text)
            else:
                # Signal / string.
                text = str(raw).strip()
                # Blank optional signal.
                if text == "":
                    # Remove optional signal keys.
                    rule.pop(key, None)
                else:
                    # Store string.
                    rule[key] = text
        # Sync clip bounds onto the spec.
        self.reward_spec.data["clip_min"] = float(self.clip_min_var.get())
        # Clip max.
        self.reward_spec.data["clip_max"] = float(self.clip_max_var.get())
        # Refresh list labels.
        self.refresh_rule_list()
        # Reselect the same index.
        self.rule_list.selection_set(self.selected_index)
        # Status.
        self.on_status(f"Applied edits to rule '{rule.get('id')}'")

    def add_rule(self):
        """Append a new blank rule of the selected type."""
        # Require a loaded spec.
        if self.reward_spec is None:
            # Warn the user.
            messagebox.showwarning("Rewards", "Load a game pack first.")
            # Abort.
            return
        # Build a typed skeleton using shared defaults.
        new_rule = default_rule_for_type(
            self.new_rule_type.get(),
            self.signal_names,
            len(self.reward_spec.rules) + 1,
        )
        # Append.
        self.reward_spec.rules.append(new_rule)
        # Refresh UI.
        self.refresh_rule_list()
        # Select the new rule.
        index = len(self.reward_spec.rules) - 1
        # Highlight.
        self.rule_list.selection_clear(0, tk.END)
        # Select.
        self.rule_list.selection_set(index)
        # Populate editor.
        self.selected_index = index
        # Build fields.
        self.build_editor_for_rule(new_rule)

    def remove_rule(self):
        """Delete the selected rule."""
        # Need selection.
        if self.selected_index is None or self.reward_spec is None:
            # Nothing to delete.
            return
        # Pop the rule.
        removed = self.reward_spec.rules.pop(self.selected_index)
        # Clear selection.
        self.selected_index = None
        # Refresh list.
        self.refresh_rule_list()
        # Clear editor.
        self.clear_editor()
        # Status.
        self.on_status(f"Removed rule '{removed.get('id')}'")

    def save_rewards(self):
        """Persist the current reward spec to the pack rewards.json."""
        # Require a spec.
        if self.reward_spec is None:
            # Warn.
            messagebox.showwarning("Rewards", "Nothing to save.")
            # Abort.
            return
        # Apply any pending editor edits first.
        if self.selected_index is not None:
            # Flush editor -> rule dict.
            self.apply_rule_edits()
        # Sync clips.
        self.reward_spec.data["clip_min"] = float(self.clip_min_var.get())
        # Clip max.
        self.reward_spec.data["clip_max"] = float(self.clip_max_var.get())
        # Write to disk.
        path = self.reward_spec.save()
        # Update training override path.
        self.vars["rewards_path"].set(path)
        # Notify.
        messagebox.showinfo("Rewards", f"Saved rewards to:\n{path}")
        # Status log.
        self.on_status(f"Saved rewards -> {path}")

    def reload_rewards(self):
        """Reload rewards/memory from the currently selected game pack."""
        # Import locally to avoid circular imports at module load.
        from pdt.game_pack import load_game_pack

        # Resolve pack id from the UI.
        game_id = self.vars["game_id"].get()
        # Load pack.
        pack = load_game_pack(game_id, games_root=self.vars["games_root"].get())
        # Reload editor state.
        self.load_from_pack(pack)

    def import_memory_file(self):
        """Import an external memory address JSON into the active pack memory."""
        # Require an active memory bank to merge into.
        if self.memory_bank is None:
            # Warn.
            messagebox.showwarning("Memory", "Load a game pack first.")
            # Abort.
            return
        # Ask the user for a JSON file.
        path = filedialog.askopenfilename(
            title="Import memory address JSON",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        # Cancelled dialog.
        if not path:
            # Abort.
            return
        try:
            # Load the imported bank.
            imported = MemoryBank.from_file(path)
            # Merge signals into the active bank.
            changed = self.memory_bank.merge_import(imported, overwrite=True)
            # Persist the merged bank to the pack memory path.
            saved = self.memory_bank.save()
            # Refresh signal names for dropdowns.
            self.signal_names = self.memory_bank.signal_names()
            # Point training at the updated memory file.
            self.vars["memory_path"].set(saved)
            # Update status strip.
            self.memory_status.set(f"MEMORY: imported ({changed} signals)")
            # Notify.
            messagebox.showinfo(
                "Memory",
                f"Imported {changed} signal(s) from:\n{path}\n\nSaved to:\n{saved}",
            )
            # Rebuild editor if a rule is selected so signal lists refresh.
            if self.selected_index is not None:
                # Rebuild with updated dropdown values.
                self.build_editor_for_rule(self.reward_spec.rules[self.selected_index])
            # Status log.
            self.on_status(f"Imported memory map ({changed} signals)")
        except Exception as exc:
            # Show parse / IO errors.
            messagebox.showerror("Memory import failed", str(exc))
