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

The algorithm that estimates f0 from audio: WORLD's DIO (fast) or Harvest (better) with StoneMask
refinement — the traditional DSP estimators — or an **ML estimator**. _Avoid_: "engine" for these.

## ML estimator

An f0 estimator backed by a neural model rather than a DSP algorithm. Its voicing comes from the
model's own confidence, not from the energy voicing gate, and it has no StoneMask stage. RMVPE and
SwiftF0 are the supported ones. _Avoid_: calling it a "resampler" or "engine".

## Model file

The weights file an ML estimator loads. RMVPE's `rmvpe.onnx` is user-supplied, looked up in the
executable's directory and then `KIRAFRQ_ML_DIR`; without it RMVPE cannot run — the GUI disables
Generate and the CLI errors — never a fallback to WORLD. SwiftF0's `swiftf0.onnx` uses the same
lookup as an override but ships bundled, so no file is needed.
_Avoid_: "weights" alone when the on-disk artifact is meant.

## Spurious voicing

Frames an f0 estimator marks voiced over breath, noise or near-silence — most visibly a low, flat f0
line through a noise floor. Recommended tuning exists to suppress this class; it is not a name
for every wrong voiced/unvoiced decision. _Avoid_: "over-voicing" in canonical prose.

## Recommended tuning

The umbrella toggle (on by default) that applies KiraFrqGen's post-estimation voicing workarounds
on top of the raw estimator output: the energy voicing gate (the WORLD pair) and the Harvest-only
aperiodicity gate. Turned off, the estimator's output is written as-is. The GUI checkbox is
"Apply recommended tuning"; the CLI opt-out is `--no-recommended-tuning`. _Avoid_: "WORLD quirks"
(the former name) and bare "tuning".

## Energy voicing gate

The workaround recommended tuning applies to spurious voicing: a frame is kept voiced only when its
per-frame amplitude clears a relative share of the file's voiced-frame loudness (KiraFrqGen's
computed amplitude, not one read from an existing frq). It targets the quiet noise-floor class —
breath and onsets at speech level pass through. _Avoid_: bare "gate" — an aperiodicity gate is a
different stage.

## Aperiodicity gate

The second workaround recommended tuning applies to spurious voicing, on Harvest only and on top of
the energy voicing gate: a frame is kept voiced only when its raw D4C LoveTrain aperiodicity
statistic clears 0.85. It targets the louder aperiodic residuals — onsets, transitions and voiced
consonants — that the energy gate's loudness test lets through. Both thresholds are internal, with
no user knob. _Avoid_: "D4C gate" or bare "gate" alone; the full term disambiguates it from the
energy voicing gate.

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

## Ensure Japanese codepage

The opt-in `desc.mrq` mode that also writes each wav's name as a Japanese-locale (CP932) system
spells it, so a table made on a non-Japanese machine works in the Japanese region: the on-disk name
is encoded with the machine's 8-bit code page and the bytes decoded as CP932. A no-op on code
page 932; while it is on, a wav counts as having an mrq table only when both names exist.

## KiraFrqGen

The bulk frequency-table generation tool of this project, shipped as the CLI `kirafrqgen-cli` and
the GUI `kirafrqgen-gui` over one core.
