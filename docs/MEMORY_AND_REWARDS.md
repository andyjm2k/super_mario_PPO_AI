# memory.json and rewards.json reference

Pyboy Deep Trainer keeps game-specific RAM and reward logic in two JSON files inside each game pack:

| File | Role |
|------|------|
| `games/<id>/memory.json` | Named signals: addresses + how to decode them |
| `games/<id>/rewards.json` | Declarative reward / terminal rules that **reference those signal names** |

Rewards never hard-code hex addresses. They only use signal names from `memory.json`, so you can retarget a new game by swapping the memory map (or importing one in the UI) without rewriting reward logic from scratch.

Full example: [`games/mario_deluxe/memory.json`](../games/mario_deluxe/memory.json) and [`games/mario_deluxe/rewards.json`](../games/mario_deluxe/rewards.json).

---

## memory.json

### Top-level shape

```json
{
  "game_id": "my_game",
  "description": "Optional human-readable note",
  "defaults": {
    "tile_digit_base": 208,
    "max_progress_tile": 12
  },
  "signals": {
    "SIGNAL_NAME": { "...": "..." }
  }
}
```

| Field | Required | Description |
|-------|----------|-------------|
| `game_id` | no | Pack id; for documentation / tooling |
| `description` | no | Free-text note |
| `defaults` | no | Shared decoder parameters (see below) |
| `signals` | **yes** | Map of signal name → definition |

Signal names are case-sensitive identifiers used in `rewards.json` (`LEVEL_PROGRESS`, `LIVES_REMAINING`, …). Prefer `UPPER_SNAKE_CASE`.

### Address formats

Addresses may be:

- hex strings: `"0xC24B"`
- decimal integers: `49739`

Multi-byte signals use `"addresses": [...]` (ordered list). Single-byte signals use `"address": "..."`.

### Decoder modes (`decode`)

Every signal needs a `decode` mode (default if omitted: `raw`).

#### `raw`

Read one byte as a float.

```json
"LIVES_REMAINING": {
  "address": "0xC17F",
  "decode": "raw",
  "note": "Remaining lives counter"
}
```

Use for counters, flags, motion markers, or any single WRAM/VRAM byte you will compare or track over time.

#### `tile_percent`

Treat a HUD tile byte as a progress segment and map it to **0–100**.

```json
"LEVEL_PROGRESS": {
  "address": "0xC24B",
  "decode": "tile_percent",
  "tile_digit_base": 208,
  "max_progress_tile": 12
}
```

Formula:

1. `segment = memory[address] - tile_digit_base` (clamped; out of range → 0)
2. `percent = (segment / max_progress_tile) * 100`

| Field | Default | Meaning |
|-------|---------|---------|
| `tile_digit_base` | pack `defaults`, else `208` | Tile id for digit / segment 0 |
| `max_progress_tile` | pack `defaults`, else `12` | Segment count that means 100% |

Best for `monotonic_progress` rules that expect a 0–100 progress signal.

#### `digit_tiles_int`

Assemble several HUD digit tiles into one integer (e.g. score).

```json
"SCORE": {
  "addresses": ["0xC103", "0xC104", "0xC105", "0xC106", "0xC107"],
  "decode": "digit_tiles_int",
  "tile_digit_base": 208,
  "max_digit": 12
}
```

| Field | Default | Meaning |
|-------|---------|---------|
| `addresses` | **required** | MSD → LSD digit tile addresses |
| `tile_digit_base` | pack defaults / `208` | Tile id for digit 0 |
| `max_digit` | `12` | Max digit offset accepted |

Leading zeros are skipped when assembling (`00123` → `123`).

#### `timer_critical`

Return `1.0` when three timer digit tiles show the pattern **001**, else `0.0`.

```json
"TIMER_CRITICAL": {
  "addresses": ["0x9811", "0x9812", "0x9813"],
  "decode": "timer_critical",
  "tile_digit_base": 208
}
```

Used as `also_signal` on a `timeout` rule so the episode ends when the in-game timer is essentially exhausted.

### Pack-level `defaults`

Values under `defaults` apply to any signal that omits the same key:

```json
"defaults": {
  "tile_digit_base": 208,
  "max_progress_tile": 12
}
```

Per-signal fields always win over `defaults`.

### Optional fields

| Field | Description |
|-------|-------------|
| `note` | Comment for humans / UI; ignored by the engine |

### Importing memory maps

In the training UI **REWARDS** tab → **IMPORT MEMORY**:

1. Choose an external `.json` with a `signals` object (same schema).
2. Signals are merged into the active pack’s `memory.json` (overwrite by name).
3. The file is saved back to the pack so reward dropdowns see the new names.

Minimal import file:

```json
{
  "signals": {
    "CUSTOM_FLAG": {
      "address": "0xC000",
      "decode": "raw",
      "note": "Discovered flag byte"
    }
  }
}
```

---

## rewards.json

### Top-level shape

```json
{
  "game_id": "my_game",
  "description": "Optional note",
  "clip_min": -50.0,
  "clip_max": 50.0,
  "rules": [
    { "id": "...", "type": "...", "enabled": true }
  ]
}
```

| Field | Required | Description |
|-------|----------|-------------|
| `clip_min` / `clip_max` | no | Final per-step reward clamp (defaults −50 / 50) |
| `rules` | **yes** | Ordered list of rule objects |
| `game_id` / `description` | no | Documentation only |

Rules run **in list order** each env step. The final scalar is clipped to `[clip_min, clip_max]`.

### Shared rule fields

Every rule supports:

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Stable name for logs / UI (`living_cost`, `death`, …) |
| `type` | string | One of the rule types below |
| `enabled` | bool | Default `true`; set `false` to disable without deleting |
| `note` | string | Optional comment |

Any `signal` / `progress_signal` / `also_signal` value **must** match a key in `memory.json` → `signals`.

### Rule types

#### `constant`

Add a fixed value every step (living cost or small bonus).

```json
{
  "id": "living_cost",
  "type": "constant",
  "enabled": true,
  "value": -0.05
}
```

| Field | Type | Description |
|-------|------|-------------|
| `value` | float | Added to the step reward |

#### `monotonic_progress`

Reward only **new furthest** values of a progress signal (anti back-and-forth farming).

```json
{
  "id": "progress",
  "type": "monotonic_progress",
  "signal": "LEVEL_PROGRESS",
  "scale": 12.0,
  "new_best_bonus": 4.0,
  "backward_penalty": -4.0,
  "no_forward_penalty": -0.2,
  "oscillation_penalty": -0.5,
  "oscillation_window": 6
}
```

| Field | Type | Description |
|-------|------|-------------|
| `signal` | string | Progress signal (usually 0–100 via `tile_percent`) |
| `scale` | float | Multiplier on newly claimed progress (`new_territory * scale`) |
| `new_best_bonus` | float | Flat bonus when the episode best increases |
| `backward_penalty` | float | Applied when progress decreases this step |
| `no_forward_penalty` | float | Applied when not setting a new best (stall / re-cross / retreat) |
| `oscillation_penalty` | float | Extra penalty when direction flips often without new territory |
| `oscillation_window` | int | How many recent non-zero direction signs to keep |

Behaviour summary:

- **New best** → progress reward + optional new-best bonus  
- **Moved backward** → backward + no-forward penalties  
- **Flat / re-cross** → no-forward penalty  
- **Wiggle without new territory** → oscillation penalty when the window is full of flips  

#### `equals_terminal`

Fire when a signal equals a sentinel (flag reached, boss defeated, …).

```json
{
  "id": "flag",
  "type": "equals_terminal",
  "signal": "FLAG_STATE",
  "equals": 80,
  "reward": 200.0,
  "terminal": true,
  "replace_reward": false
}
```

| Field | Type | Description |
|-------|------|-------------|
| `signal` | string | Signal to read |
| `equals` | float | Target value |
| `reward` | float | Bonus / penalty when matched |
| `terminal` | bool | End the episode (`terminated=True`) |
| `replace_reward` | bool | If `true`, replace the whole step reward; else add |

#### `decrement_terminal`

Fire when a counter **drops** vs the previous sample (lives lost, HP decreased).

```json
{
  "id": "death",
  "type": "decrement_terminal",
  "signal": "LIVES_REMAINING",
  "reward": -40.0,
  "terminal": true,
  "replace_reward": true
}
```

| Field | Type | Description |
|-------|------|-------------|
| `signal` | string | Counter signal (`raw` decode) |
| `reward` | float | Usually a large negative penalty |
| `terminal` | bool | End the episode |
| `replace_reward` | bool | Typically `true` so death dominates other shaping |

Important: do **not** treat a fixed lives value (e.g. `4`) as death. Starting an episode with 4 lives is normal; only a **decrement** counts.

#### `stuck`

Penalise (and optionally truncate) when a motion / position signal stops changing.

```json
{
  "id": "stuck",
  "type": "stuck",
  "signal": "MARIO_MOTION",
  "penalty": -0.1,
  "grace_steps": 100,
  "limit_steps": 800,
  "truncate": true
}
```

| Field | Type | Description |
|-------|------|-------------|
| `signal` | string | Motion marker that should change when the agent moves |
| `penalty` | float | Per-step penalty while stuck (after grace) |
| `grace_steps` | int | Episode steps before stuck detection starts |
| `limit_steps` | int | Consecutive stuck steps before truncation |
| `truncate` | bool | Soft-end the episode when the limit is hit |

#### `timeout`

End the episode when a step budget is exhausted, and/or when an optional critical signal fires.

```json
{
  "id": "timeout",
  "type": "timeout",
  "base_steps": 2500,
  "max_steps": 12000,
  "progress_signal": "LEVEL_PROGRESS",
  "also_signal": "TIMER_CRITICAL",
  "penalty": -20.0,
  "truncate": true
}
```

| Field | Type | Description |
|-------|------|-------------|
| `base_steps` | int | Step budget at 0% progress |
| `max_steps` | int | Step budget at 100% progress |
| `progress_signal` | string | Optional 0–100 signal that scales the budget |
| `also_signal` | string | Optional signal; values ≥ `0.5` force timeout (e.g. `timer_critical`) |
| `penalty` | float | Added when timeout fires (if not already terminated) |
| `truncate` | bool | Soft-end (`truncated=True`) rather than success/fail terminate |

Budget interpolation:

`threshold = base_steps + (max_steps - base_steps) * (progress_pct / 100)`

---

## How the two files connect

```
memory.json                         rewards.json
───────────                         ────────────
LEVEL_PROGRESS (tile_percent)  ←──  monotonic_progress.signal
LIVES_REMAINING (raw)          ←──  decrement_terminal.signal
FLAG_STATE (raw)               ←──  equals_terminal.signal
MARIO_MOTION (raw)             ←──  stuck.signal
TIMER_CRITICAL (timer_critical)←──  timeout.also_signal
```

Workflow for a new game:

1. Discover addresses (debugger / dumps) and write `memory.json` signals.  
2. Point reward rules at those names in `rewards.json`.  
3. Or: keep an existing rewards profile and **IMPORT MEMORY** so signal names match.  
4. Tune coefficients in the UI **REWARDS** tab → **SAVE REWARDS**.

---

## Minimal starter templates

### Minimal `memory.json`

```json
{
  "game_id": "my_game",
  "signals": {
    "PROGRESS": {
      "address": "0xC000",
      "decode": "raw",
      "note": "Replace with a real progress byte or tile_percent signal"
    },
    "LIVES": {
      "address": "0xC001",
      "decode": "raw"
    },
    "MOTION": {
      "address": "0xC002",
      "decode": "raw"
    }
  }
}
```

### Minimal `rewards.json`

```json
{
  "clip_min": -50.0,
  "clip_max": 50.0,
  "rules": [
    {
      "id": "living_cost",
      "type": "constant",
      "value": -0.05
    },
    {
      "id": "progress",
      "type": "monotonic_progress",
      "signal": "PROGRESS",
      "scale": 1.0,
      "new_best_bonus": 1.0,
      "backward_penalty": -1.0,
      "no_forward_penalty": -0.1,
      "oscillation_penalty": -0.2,
      "oscillation_window": 6
    },
    {
      "id": "death",
      "type": "decrement_terminal",
      "signal": "LIVES",
      "reward": -40.0,
      "terminal": true,
      "replace_reward": true
    },
    {
      "id": "stuck",
      "type": "stuck",
      "signal": "MOTION",
      "penalty": -0.1,
      "grace_steps": 100,
      "limit_steps": 800,
      "truncate": true
    },
    {
      "id": "timeout",
      "type": "timeout",
      "base_steps": 2500,
      "max_steps": 2500,
      "penalty": -20.0,
      "truncate": true
    }
  ]
}
```

---

## Tips

- Prefer **decoded** progress (`tile_percent` → 0–100) for `monotonic_progress`; raw tiles work but scales feel arbitrary.  
- Prefer **decrement** for lives/HP; equality checks on a lives count false-trigger on mid-run starts.  
- Keep `id` values unique; the UI and logs key components by id.  
- Disable a rule with `"enabled": false` while experimenting instead of deleting it.  
- After hand-editing JSON, use **RELOAD** in the REWARDS tab (or restart the UI) so the editor picks up disk changes.  
- Invalid signal names raise at runtime when the env steps — fix the name in either file so they match exactly.
