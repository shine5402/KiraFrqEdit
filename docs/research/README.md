# Research notes

Groundwork research for KiraFrqEdit, collected 2026-09. Every claim is either:

- **verified** against real files from a local UTAU installation (see `../../AGENTS.local.md` for machine-local state), and/or
- **cited** from upstream sources (code, manuals, author blogs) listed in each document.

## Documents

| File | Contents |
| --- | --- |
| [frq-format.md](frq-format.md) | UTAU `frq` binary format: verified layout, unvoiced/volume semantics, robustness rules |
| [mrq-format.md](mrq-format.md) | moresampler `desc.mrq` cache: verified against Kanru Hua's `mrq.c`/`mrq.h` |
| [pmk-format.md](pmk-format.md) | TIPS `pmk`: reverse-engineered, corpus-verified; no public spec exists |
| [other-formats.md](other-formats.md) | `vs4ufrq`, `frc`, `dio`+`star`+`platinum`: behavior + engine matrix, not yet byte-researched |
| [f0-generation.md](f0-generation.md) | WORLD (DIO/Harvest), SpeedWagon, Rust integration options |
| [world-integration.md](world-integration.md) | Vendored WORLD v1.0.1 build on MSVC and clang-cl, `cc` + raw `extern "C"` shim, timings, Harvest pure-tone gotcha, license duties |
| [harvest-voicing.md](harvest-voicing.md) | Harvest's voiced/unvoiced decision in the vendored source, the constants behind it, ranked levers to stop it voicing noise (D4C gate, duration knobs, DIO), upstream design intent |
| [ml-f0-estimators.md](ml-f0-estimators.md) | Permissive ML f0 options (RMVPE, CREPE, FCPE, SwiftF0): weights/licenses, Rust runtimes (`ort` vs `tract`), CPU cost vs WORLD, hop-256/71-800 Hz mapping, U/V evidence |
| [swiftf0-adoption.md](swiftf0-adoption.md) | SwiftF0 as a second ML estimator: `ort` (ORT 1.28.0) runs the committed ONNX bit-exactly, MIT weight provenance, integration delta vs the RMVPE path, public-evidence gaps |
| [frqeditor-manual-notes.md](frqeditor-manual-notes.md) | Digest of the frqeditor manual (UTAU周波数表資料 0.90): concepts, engine behavior, editing + auto check/fix spec |
| [frame-conventions.md](frame-conventions.md) | Frame grid, count and edge conventions across `frq`/`pmk`/`mrq`, derived from the local corpus |
| [frq-amplitude-scale.md](frq-amplitude-scale.md) | Amplitude array scales per writer, and the formula KiraFrqGen should write |
| [moresampler-llsm.md](moresampler-llsm.md) | moresampler `.llsm` cache semantics, staleness rules and the deletion policy |
| [wav-decode-resample.md](wav-decode-resample.md) | Decode + resample crate choices (`hound`, `rubato`) with the rejected alternatives |

## Primary sources

- `.local/UTAU周波数表資料.pdf` (machine-local, gitignored) — frqeditor's manual, 278 pages, 0.90
  (2015-05-23), by ちていこ. It documents the whole UTAU frequency-table ecosystem and contains the
  original spec proposal for frqeditor's check/repair feature (chapter 5).
- Machine-local extraction: `.local/manual.extract.txt` (gitignored), regenerable with:

  ```powershell
  python -m pip install pypdf fonttools
  python .local/extract.py   # reads .local/UTAU周波数表資料.pdf, writes .local/manual.extract.txt
  ```

  Note for agents: models here cannot read PDF attachments directly; always use the extracted text.

## Licensing / clean-room notes

- **frqeditor** is proprietary, no source. All behavior notes about it come from its manual and
  black-box observation. Do not link or decompile it into this project.
- **`mrq.c`/`mrq.h`** (github.com/Sleepwalking/mrq) are under a 3-clause BSD-style license. Reading
  them as a spec is fine; if code is ever vendored, keep the copyright notice.
- **TIPS** has no public source; the `pmk` spec here is reverse-engineered from sample files.
- **WORLD** is modified BSD (no patents claimed). Good to vendor/build.
- **SpeedWagon** source zip was published by the author (Custom.Maid) but the license is unknown;
  treat as reference only, do not copy code.
- **Engine binaries** (`moresampler.exe`, `TIPS.exe`, ...) are proprietary. Use them only for
  black-box comparison during development; never redistribute.
- **`frqeditor`'s check/fix algorithm spec** (chapter 5 of the manual) is the author's public spec
  proposal; reimplementing it is intended use.
