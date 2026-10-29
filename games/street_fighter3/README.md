# Street Fighter Alpha 3 (GBA) — mGBA pack

This pack uses **mGBA** (`"emulator": "mgba"`) so training can show **live**
frames and load a **full** savestate (CPU + PPU + memory).

PyBoyAdvance is still in the repo for other GBA packs, but it cannot do this:
`.pba_state` only restores EWRAM/IWRAM and freezes an RGB preview. Those files
are **not compatible** with mGBA. `states/versus_start.pba_state` is ignored.
Until you save a new `.mgba_state`, episodes **boot the ROM**
(`allow_boot_start: true`, `prefer_boot_start: false`).

## Important about the GameFAQs link

[FAQ #13943](https://gamefaqs.gamespot.com/gba/516513-street-fighter-alpha-3/faqs/13943) is the **Dreamcast VMU hacking guide**, not a GBA RAM map (same FAQ id is filed under the Dreamcast game).

GBA addresses used here come from the **CodeBreaker / CMGSCC** map documented in:

- [Slateman SFZ3U Cheats FAQ](https://slateman.net/faqs/sfz3u.txt) (Alpha 3 / Zero 3 Upper GBA)
- Matching USA CodeBreaker listings (e.g. Almar’s Guides)

CodeBreaker `32AAAAAA VV` → 8-bit peek at GBA address `0x0AAAAAAA` (EWRAM), e.g. `32000818` → `0x02000818`.

## Files

| File | Role |
|------|------|
| `rom/Street Fighter Alpha 3 (USA).gba` | ROM |
| `rom/gba_bios.bin` | GBA BIOS |
| `memory.json` | P1 / Opp1 / Opp2 health, ISM gauges, unlock bytes |
| `rewards.json` | Damage dealt/taken deltas + KO terminals |
| `game.json` | mGBA + fighting action set |
| `states/*.mgba_state` | Full mGBA savestates (you must record these) |

## Signals (1v1 focus)

| Signal | Address | Meaning |
|--------|---------|---------|
| `P1_HEALTH` | `0x02000818` | Your HP (~`0x90` full) |
| `P1_ISM` | `0x0200090E` | Your ISM / super gauge |
| `OPP1_HEALTH` | `0x02004428` | Opponent 1 HP |
| `OPP1_ISM` | `0x0200451E` | Opponent 1 ISM |
| `OPP2_HEALTH` | `0x02000268` | Opponent 2 (Dramatic mode) |
| `P1_ROUND_WON` | `0x02000916` | Round-won flag |

Rewards use **Opp1** as the primary foe. Enable Dramatic-mode Opp2 shaping later if needed.

## Install mGBA on Windows

`pip install mgba` and `pip install pygba` do **not** ship Windows wheels.
This project expects [hanzi/libmgba-py](https://github.com/hanzi/libmgba-py)
(mGBA **0.10.2**, MPL 2.0), which adds `save_state() -> bytes` and `load_state(vfile)`.

The bindings are imported lazily. If they are missing, creating the backend
raises an `ImportError` with these same steps. Do not assume `import mgba`
works until you have built it into the venv.

1. Install **Visual Studio 2022** (Community is enough) with the
   **Desktop development with C++** workload, and install **Git**.
2. Clone the bindings somewhere outside this repo:

   ```powershell
   git clone https://github.com/hanzi/libmgba-py.git
   ```

3. `vcvars64.bat` initializes the C++ compiler (`cl`) and **does not** add
   `MSBuild.exe`. `build_win64.bat` exits immediately if `where msbuild`
   fails. After `vcvars64`, prepend the MSBuild `Bin` directory (find it
   with `vswhere` if it is not in the usual Community path):

   ```bat
   call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
   set "PATH=C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin;%PATH%"
   where msbuild
   ```

   `where msbuild` must print `...\MSBuild\Current\Bin\MSBuild.exe` before
   you continue. The **x64 Native Tools Command Prompt for VS 2022** often
   already includes that directory; a plain `vcvars64.bat` call does not.
4. Activate this repo's venv and build (needs several GB and a long vcpkg
   compile; expect 20+ minutes). mGBA **0.10.2** does not compile against
   current vcpkg FFmpeg (`sample_fmts` was removed from `AVCodec`). The
   training backend does not record video or run Lua scripts, so the clone's
   `build_win64.bat` cmake line must include `-DUSE_FFMPEG=OFF -DUSE_LUA=OFF`
   (current vcpkg Lua also fails to link into `mgba.dll`):

   ```bat
   call C:\Users\andyj\super_mario_PPO_AI\venv\Scripts\activate.bat
   cd /d C:\path\to\libmgba-py
   build_win64.bat
   ```

   A few-second exit is a failure (usually `msbuild cannot be found in PATH`).
   `Done!` only counts after MSBuild and `setup.py` both succeed. If the
   linker says it cannot open `mgba.lib`, the C library did not build.
5. Copy the built package into the venv (DLLs must sit next to the package):

   ```bat
   xcopy /E /I /Y build\win64\mgba C:\Users\andyj\super_mario_PPO_AI\venv\Lib\site-packages\mgba
   ```

6. Confirm:

   ```bat
   python -c "import mgba; print('mgba ok')"
   ```

## Record a full savestate

After `import mgba` works:

```powershell
python tools/save_mgba_state.py --game street_fighter3 --name versus_start
```

Controls: **A/S** = A/B, **arrows** = move, **Q/W** = L/R, **Enter** = Start,
**Backspace** = Select. Get into a fight (or training), then press **F5**.
**Esc** quits. The window shows the **live** mGBA framebuffer.

F5 writes `states/versus_start.mgba_state` (CPU + PPU + memory). Training
loads that file on each episode and keeps reading live frames from
`get_rgb_frame()`. The old `tools/save_pba_state.py` tool is unchanged and
still writes `.pba_state` for PyBoyAdvance packs only.

For title-screen episodes instead, set `"prefer_boot_start": true` in
`game.json` (on-disk savestates are ignored).

KO terminals use `min_episode_length: 90` so HP=`0` on menus does not instantly end episodes.

## Train

```powershell
python training_ui.py
```

Select **street_fighter3** in GAME, then TRAIN. With no `.mgba_state` yet,
episodes boot the ROM (you will see the BIOS/title path, not a frozen
`.pba_state` preview). After F5, episodes resume from the match and the
training view follows the live framebuffer. Tune scales on the REWARDS tab.
