# GBA template pack (PyBoyAdvance)

This pack is a **scaffold** for Game Boy Advance titles using
[PyBoyAdvance](https://github.com/williamckha/PyBoyAdvance).

## Setup

1. Install the optional engine:

```bash
pip install pyboy-advance
```

2. Place files under this pack:

| Path | Purpose |
|------|---------|
| `rom/game.gba` | Your GBA ROM (rename + update `game.json` `rom`) |
| `rom/gba_bios.bin` | GBA BIOS (required). Open BIOS: [Nebuleon/ReGBA](https://github.com/Nebuleon/ReGBA/blob/master/bios/gba_bios.bin) |
| `states/*.pba_state` | Optional PDT snapshots (see below) |

3. Edit `memory.json` addresses for your game, then tune `rewards.json`.

4. Select **gba_template** (or your renamed pack) in the TRAIN / RUNNER **GAME** dropdown.

## game.json emulator fields

```json
{
  "emulator": "pyboy_advance",
  "bios": "rom/gba_bios.bin",
  "skip_bios": true,
  "allow_boot_start": true,
  "state_extensions": [".pba_state"]
}
```

- `emulator`: `pyboy` (GB/GBC) or `pyboy_advance` (GBA)
- `bios`: path relative to the pack root (GBA only)
- `skip_bios`: skip BIOS animation for faster starts
- `allow_boot_start`: train from ROM entry when no snapshots exist

## Save states

Upstream PyBoyAdvance does **not** expose PyBoy-compatible `.state` files yet.
Pyboy Deep Trainer uses a custom `*.pba_state` pickle (EWRAM + IWRAM + RGB
preview; CPU regs are saved but not restored by default — restoring them can
crash). Load path is **boot-then-apply**: warm a few frames, inject RAM, freeze
UI on `rgb_frame`.

Create one from a live backend:

```python
from pdt.emulator.factory import create_emulator_backend
from pdt.emulator.pyboy_advance_backend import save_pdt_snapshot

emu = create_emulator_backend(
    "pyboy_advance",
    rom_path="games/gba_template/rom/game.gba",
    bios_path="games/gba_template/rom/gba_bios.bin",
    skip_bios=True,
    emulation_speed=0,
)
emu.tick(120)
save_pdt_snapshot(emu, "games/gba_template/states/start.pba_state")
emu.stop()
```

With `allow_boot_start: true` you can train without snapshots (episodes start at ROM boot).

## Notes

- Native GBA resolution is **240×160**; the pack defaults obs to 120×80.
- Button names in `actions`: `a`, `b`, `left`, `right`, `up`, `down`, `start`, `select`, `l`, `r`.
- PyBoyAdvance is experimental; some games may not run correctly yet.
