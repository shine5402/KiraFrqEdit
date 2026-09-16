# f0 generation: WORLD, SpeedWagon, Rust integration

Status: research notes, no implementation yet. This is the plan for KiraFrqEdit's bulk f0
generation feature (what SpeedWagon does for UTAU users today).

## WORLD vocoder

- Official repo: <https://github.com/mmorise/World> — **modified BSD license**, "no patent in all
  algorithms" per README. Latest release v1.0.1 (2026-02).
- DIO (fast f0) and Harvest (slower, better for speech) with StoneMask refinement are the relevant
  stages; both are C++ with C-callable API:
  - `Dio(x, x_length, fs, &option, temporal_positions, f0)` / `InitializeDioOption` /
    `GetSamplesForDIO`
  - `Harvest(...)`, `StoneMask(x, x_length, fs, temporal_positions, f0, f0_length, refined_f0)`
- Defaults: `frame_period = 5 ms`, f0 floor 71 Hz, ceiling 800 Hz (StoneMask floor 40).
- Frames land on a 5 ms grid; UTAU `frq`/`mrq` want 256-sample hops at 44.1 kHz (5.805 ms), so a
  resample/interpolation step is needed when writing tables.

## SpeedWagon

- Author: Custom.Maid (<https://custom-made.seesaa.net/>). Latest known build 2013-01-10, binary
  and **source zips** published at the time (`Speedwagon130110.zip`, `Speedwagon130110SRC.zip`).
  Download host (axfc) was unreachable/HTTP 500 during research; official site is the best lead.
- No license text found -> treat source as reference-only, no code copying.
- Usage: `speedwagon_DandD.exe` with drag-and-dropped WAVs (batch, console), or `speedwagon.exe`
  configured as UTAU's "tool 2"/resampler for the built-in batch frequency-table generation.
  Outputs `_wav.frq` next to the wav. It cannot synthesize audio.
- Algorithm: unclear. The author's 2013 post calls the method roughly zero-crossing-based; a 2015
  post says a WORLD 0.2 (DIO) version was built but never released and was better. Community wiki
  flatly states it uses WORLD. For KiraFrqEdit we do not need to clone it: use WORLD DIO (and offer
  Harvest) directly, which is at least as good and lets us expose parameters.
- Useful for compatibility tests: compare KiraFrqEdit-generated tables against this binary if a
  copy can be obtained.

## Rust integration options

| Option | License | State | Verdict |
| --- | --- | --- | --- |
| Vendor WORLD C++ + `cc`/`cxx` FFI shim | BSD-3 | stable upstream | **Preferred**: pin a version, build from source, wrap `Dio`/`Harvest`/`StoneMask` in a safe Rust API |
| [`world-rs`](https://github.com/JamesFysh/world-rs) (pure Rust port, crates.io 0.1.0) | BSD-3 | brand new, untested | Watch, don't depend on it yet |
| [`Rust-WORLD`](https://github.com/BiggieBoo18/Rust-WORLD) (`rsworld`/`rsworld-sys`) | MIT | dormant since 2019 | Fallback if vendoring is painful |
| SpeedWagon itself | unknown | binary/source, XP-era toolchain | Reference only |

Practical notes for the eventual implementation:

- Keep all `unsafe`/FFI in one crate/module (`world-sys` + safe wrapper), matching repo policy once
  written.
- Input normalization: decode WAV (e.g. `hound`/`symphonia` in Rust), downmix to mono, resample to
  44.1 kHz if needed, then run DIO/Harvest.
- Map WORLD's 5 ms frame grid onto the 256-sample grid; f0 below the floor or unvoiced -> `0.0` in
  `frq`; amplitude per frame can be computed from the waveform (RMS/peak), remembering frq's
  amplitude is display-only.
- Batch UX target: select voicebank folder(s), choose engine format (frq first), skip/overwrite
  policy, progress per file; later, the check/repair pass (see manual notes) can run after
  generation.

## Sources

- <https://github.com/mmorise/World> and releases
- <https://custom-made.seesaa.net/article/302247780.html>, `/article/312531314.html`,
  `/article/423445986.html` (SpeedWagon releases and author statements)
- <https://w.atwiki.jp/utaou/pages/36.html> (community wiki claiming WORLD usage)
- crates.io: `world-rs`, `rsworld`/`rsworld-sys`
