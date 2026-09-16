# Context: KiraFrqEdit

Glossary of the project's ubiquitous language. Canonical terms only; implementation and specs live
under `docs/`.

## Frequency table

Per-wav cache of pitch information that UTAU resamplers consume so they do not re-analyze the audio.
Formats: `frq` (UTAU classic: f0 + amplitude + key), `pmk` (TIPS: f0 as period codes), `mrq`
(moresampler: f0 only, stored as one `desc.mrq` per folder). _Avoid_: "pitch file".

## Key frequency

The recording pitch of a wav, in Hz, stored in most table formats and used by UTAU's `mod`
processing. Not the same as a note's pitch. frqeditor sets it to the mean of voiced frames.
_Avoid_: calling it "average f0" in mixed company.

## f0 estimator

The algorithm that estimates f0 from audio: WORLD's DIO (fast) or Harvest (better), with StoneMask
refinement. _Avoid_: "engine" for these.

## Resampler

A UTAU synthesis engine (moresampler, TIPS, fresamp, resampler, ...). "Engine" alone is ambiguous
in this project; prefer "resampler".

## Voicebank

A folder tree of wav recordings plus metadata (`oto.ini`, `prefix.map`). KiraFrqGen only recurses it
for wavs; it does not interpret the metadata.

## Sidecar

A generated artifact stored next to its wav or folder: `<name>_wav.frq`, `<name>_wav.pmk`, a
folder's `desc.mrq`, moresampler's `.llsm`.

## llsm

moresampler's per-wav DSP model cache, derived from audio and f0; stale whenever f0 changes, and
regenerated on demand.

## KiraFrqGen

The bulk frequency-table generation tool of this project: CLI `kira-frqgen` and GUI `KiraFrqGen`
sharing one core.
