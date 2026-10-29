# Pyboy Deep Trainer

Generic PyBoy RL trainer (Stable-Baselines3 / sb3-contrib) with **per-game packs**,
**declarative memory maps**, and **declarative reward rules** editable in the UI.

## Emulator engines

| Engine | `game.json` value | System | Package |
|--------|-------------------|--------|---------|
| [PyBoy](https://github.com/Baekalfen/PyBoy) | `"emulator": "pyboy"` (default) | GB / GBC | `pyboy==2.6.0` |
| [PyBoyAdvance](https://github.com/williamckha/PyBoyAdvance) | `"emulator": "pyboy_advance"` | GBA | `pyboy-advance` |
| [mGBA](https://mgba.io/) via [hanzi/libmgba-py](https://github.com/hanzi/libmgba-py) | `"emulator": "mgba"` | GBA | no Windows wheel; build `libmgba-py` (see `games/street_fighter3/README.md`). `vcvars64.bat` does not add MSBuild — prepend `MSBuild\Current\Bin` |

GBA packs also need a BIOS (`"bios": "rom/gba_bios.bin"`). See `games/gba_template/README.md`.

Street Fighter Alpha 3 uses mGBA. A `.pba_state` file is **not** an mGBA savestate (it only restores EWRAM/IWRAM plus a frozen preview). After `import mgba` works, re-save with `python tools/save_mgba_state.py --game street_fighter3 --name versus_start` (F5 writes `states/<name>.mgba_state`). Until that file exists, episodes boot the ROM.

## Game packs

Each game lives under `games/<id>/` so ROMs and states never co-mingle:

```
games/<id>/
  game.json       # rom path, actions, obs size
  memory.json     # named RAM/VRAM signals
  rewards.json    # declarative reward rules
  rom/            # .gbc / .gb ROM (+ .ram)
  states/         # PyBoy .state start snapshots
```

Mario Deluxe ships as `games/mario_deluxe/`.

## Declarative rewards

Reward logic is JSON, not hard-coded Python. Rule types:

| Type | Purpose |
|------|---------|
| `constant` | Per-step living cost / bonus |
| `monotonic_progress` | Reward new furthest progress; punish retreat / wiggle |
| `equals_terminal` | Terminal when a signal equals a value (flag) |
| `decrement_terminal` | Terminal when a counter drops (lives) |
| `stuck` | Penalty + truncate when a motion signal freezes |
| `timeout` | Progressive step budget (+ optional critical timer signal) |

Edit rules in the **REWARDS** tab of the training UI. Use **IMPORT MEMORY** to merge an external `memory.json` address map into the active pack.

Full schema reference: [docs/MEMORY_AND_REWARDS.md](docs/MEMORY_AND_REWARDS.md).

## Quick start

```bash
pip install -r requirements.txt

# Pyboy Deep Trainer UI (TRAIN / MODEL / REWARDS tabs)
python training_ui.py

# headless train
python train_model.py

# model runner
python runner_ui.py

# CLI evaluate
python run_model.py --model train/best_progress_model.zip --game mario_deluxe
```

PyBoy is pinned to **2.6.0** (existing `.state` files are incompatible with 2.7).

## Tests

```bash
python -m unittest discover -s tests -v
```
