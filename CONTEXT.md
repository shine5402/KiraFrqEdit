# Context: KiraFrqEdit

Glossary of the project's ubiquitous language. Canonical terms only; implementation and specs live
under `docs/`.

## Frequency table

Per-wav cache of pitch information that UTAU resamplers consume so they do not re-analyze the audio.
Formats: `frq` (UTAU classic: f0 + amplitude + key), `pmk` (TIPS: f0 as period codes), `mrq`
(moresampler: f0 only, stored as one `desc.mrq` per folder). _Avoid_: "pitch file".

## Amplitude (volume)

The per-frame loudness value stored alongside f0 in `frq`; a human-facing reference, drawn next to
the f0 curve by UTAU's and frqeditor's table editors, used to judge whether f0 detection is sane
where it should or should not be. Not synthesis input; scale is writer-dependent. _Avoid_: bare
"volume" when the stored number is meant.

## Key frequency

The recording pitch of a wav, in Hz, stored in most table formats and used by UTAU's `mod`
processing. Not the same as a note's pitch. frqeditor sets it to the mean of voiced frames.
_Avoid_: calling it "average f0" in mixed company.

## Voiced frame

A frame that carries pitch: `f0 > 0.0` in `frq`/`mrq`; `pmk` spells it `code != 49`. Key frequency
and `pmk`'s average code are means over voiced frames/entries. _Avoid_: "sounded frame" — unvoiced
consonants are loud but pitchless.

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

moresampler's per-wav DSP model cache, derived from audio and the f0 moresampler uses (its own
estimate, corrected by `desc.mrq`); regenerated on demand.

## Sharing flag

The opt-in `desc.mrq` mode that also writes each wav's Japanese-side name, so a table made on a
non-Japanese machine works in the Japanese region: the on-disk name is encoded with the machine's
8-bit code page and the bytes decoded as CP932. A no-op on code page 932; with the flag on, a wav
counts as having an mrq table only when both names exist.

## KiraFrqGen

The bulk frequency-table generation tool of this project: CLI `kira-frqgen` and GUI `KiraFrqGen`
sharing one core.
