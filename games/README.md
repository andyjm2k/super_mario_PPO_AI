# Game packs

Drop one folder per game here. Each pack is self-contained:

- `game.json` — ROM path, actions, observation crop/size
- `memory.json` — named memory signals (addresses + decoders)
- `rewards.json` — declarative reward rules
- `rom/` — ROM file(s)
- `states/` — PyBoy `.state` start snapshots

See `mario_deluxe/` for a complete GB/GBC example and `gba_template/` for a
PyBoyAdvance (GBA) scaffold.

**Schema docs:** [docs/MEMORY_AND_REWARDS.md](../docs/MEMORY_AND_REWARDS.md) — how to structure `memory.json` and `rewards.json`.

**Emulators:** set `"emulator": "pyboy"` (GB/GBC) or `"emulator": "pyboy_advance"` (GBA) in `game.json`.
GBA packs need a BIOS and use `*.pba_state` snapshots (or `allow_boot_start`) — see `gba_template/README.md`.
