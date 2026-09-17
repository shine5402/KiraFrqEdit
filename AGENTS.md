# KiraFrqEdit

KiraFrqEdit is an open-source recreation of frqeditor: a tool to inspect, edit, check and
batch-generate UTAU frequency tables (`frq` and related formats). Frequency tables store the f0
contour and voiced/unvoiced information used by resampler engines; when they are wrong, UTAU
outputs wrong pitch, glitches and pops.

## Goal and scope

- Native read/write/edit for UTAU `frq` (f0 + volume + key frequency).
- frqeditor-parity automatic check/repair for half/double-pitch errors and unvoiced holes,
  without its quirks (spec: `docs/research/frqeditor-manual-notes.md`).
- Batch f0 generation with WORLD (DIO/Harvest), replacing SpeedWagon-style workflows.
- Conversion/interop across engine formats: `frq`, `pmk`, `mrq`; best effort `vs4ufrq`, `frc`,
  `dio`/`star`/`platinum` (milestone 2+, see `docs/research/other-formats.md`).
- CLI for batch operations plus an egui GUI.

## Planned stack

- Rust workspace: core library crate, CLI binary, egui/eframe GUI.
- C/C++ only where necessary (WORLD), built from vendored source via `cc`/`cxx`, wrapped in a safe
  Rust API; keep FFI/`unsafe` confined to one module. Landed as `crates/kirafrq-world-binding` (vendored
  WORLD v1.0.1), verified on MSVC and clang-cl.
- File formats implemented clean-room from `docs/research/`; no code from proprietary tools.

## Status

- 2026-09: research phase done; workspace scaffold landed (#15): five crates under `crates/`,
  WORLD vendored under `third_party/World`, per-platform compiler CI (clang-cl / Apple clang /
  gcc) and tag-triggered CD under
  `.github/workflows/`. The core formats, WAV normalization, the generation pipeline, the CLI and
  the GUI (#23) have landed; next is end-to-end validation (#22).
- Research results live in `docs/research/` (README is the index); format specs were verified
  against real files from the local UTAU corpus where possible.

## Repo layout

- `crates/` — the Cargo workspace: `kirafrq-formats` (neutral `FrequencyTable` + frq/pmk/mrq IO),
  `kirafrq-audio` (WAV decode and mono 44.1 kHz normalization), `kirafrq-world-binding` (vendored
  WORLD wrapper), `kirafrqgen-core` (generation pipeline), `kirafrqgen-cli` / `kirafrqgen-gui`
  (the CLI and GUI binaries).
- `third_party/World` — vendored WORLD v1.0.1 sources, built by `kirafrq-world-binding`'s `cc` build.
- `.github/workflows/` — CI with one compiler per platform (Windows clang-cl, macOS Apple
  clang, Linux gcc); CD builds binaries on version tags the same way.
- `docs/research/` — research notes and verified format specs; read before touching formats.
- `docs/agents/` — agent workflow conventions (issue tracker, triage labels, domain docs).
- `docs/adr/` — architecture decision records, created as decisions are made.
- `AGENTS.local.md` — machine-local state (paths, corpus, tooling, `.local/` artifacts such as the
  frqeditor manual PDF and its extracted text). Gitignored, not auto-injected by opencode, so check
  it when machine specifics matter.

## Local data hygiene

This repo is public. Never commit machine-local or corpus-identifying details — voicebank
folder/file names, per-bank paths, corpus or install paths — into any tracked file or issue.
Those live in `AGENTS.local.md`; tracked docs describe the corpus generically ("the local
corpus", "one bank"), keeping evidence as counts and behaviors rather than names.

## Agent skills

### Issue tracker

Issues and specs live as GitHub issues in this repo, managed with the `gh` CLI; PRs are not a triage surface. See `docs/agents/issue-tracker.md`.

### Triage labels

The five canonical roles map to their default label strings (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `CONTEXT.md` plus `docs/adr/` at the repo root. See `docs/agents/domain.md`.
