# frqeditor manual digest (UTAU周波数表資料 0.90)

The manual PDF (`.local/UTAU周波数表資料.pdf`, machine-local and gitignored) is frqeditor's user
manual (278 pages, 2015-05-23, by ちていこ, covering frqeditor 1.17). It documents UTAU frequency
tables generally, the generation/repair workflow, and — in chapters 4–5 — an explicit algorithm
spec for frqeditor's automatic check/repair feature. Line numbers below refer to
`.local/manual.extract.txt`.

This file is the feature/algorithm reference for reimplementing frqeditor. Formats are in the
sibling docs; page numbers are the manual's own.

## 1. Domain concepts

- A frequency table stores the **f0 (pitch) array + key frequency + volume** for a wav sample,
  so resamplers don't have to re-estimate pitch per synthesis. Saved next to the wav with an
  engine-specific extension (`frq`, `pmk`, ...). (2-2)
- **Key frequency (キー周波数)**: stored in most formats; used for `mod` processing:
  additional pitch shift ≈ `(f0 - key) * mod / 100`. Wrong key only matters when `mod != 0`;
  it does not cause glitches itself. `mod` range -200..200. (2-4)
- **Unvoiced ("周波数なし", shown as `----`)**: absence of f0. Normal for silence/low level, but
  also the symptom of estimation failure inside voiced regions. Engine behavior differs, see
  `other-formats.md`. Guideline: never leave voiced regions unvoiced; leave true unvoiced
  consonants/breaths unvoiced. (2-5)
- **Corruption ("周波数表の破綻")** = anything making UTAU output out-of-tune or noisy; the two
  targeted classes are **half/double pitch errors** (x2, x3, x4, x1.5, x1/2, x2/3 - regular
  ratios from harmonic mis-detection) and **unvoiced holes**. Other irregular mis-estimations
  are out of scope for auto-repair. (4-1)
- Low-pitch quirk: f0 <= ~172 Hz means one 256-sample volume bar is shorter than a period, so
  amplitude alternates large/small. Not a defect. (4-1-5)

## 2. Generation and conversion workflows

- Generation methods: UTAU original-settings batch (frq), frqeditor batch, SpeedWagon, at first
  synthesis, or fixed-frequency tables (replace samples with a generated sine at the recording
  pitch, generate, restore). (3-3..3-8)
- Conversion: frqeditor converts dio/frc/frq/vs4ufrq/pmk both ways; conversions are lossy, so
  convert once after editing instead of round-tripping. `pmk -> frq` pulls volume from the wav.
  (3-9)
- Regeneration hazards: engines regenerate derived tables when versions change (TIPS on version
  change; moresampler by timestamps; VS4U <1.4 always regenerates unless made by frqeditor >=1.16
  / VS4U >=1.41). Distribution guidance in 2-2/3-10.
- Repair priority order (4-2-4): 1) re-generate with another engine, 2) regenerate after source
  processing (EQ/denoise), 3) tool auto check/repair, 4) manual editor repair, 5) copy from pitch
  software, 6) manual edit from waveform analysis. Manual edits should be a last resort.
- Source processing constraints: delete silence and any pitch-changing processing before
  generating; EQ/denoise before generation if it doesn't touch the f0 region. (4-2-3)

## 3. frqeditor editor: feature spec (3-13)

- Form selection: Tools > Options > pick engine; the table format follows the selected engine
  (engines without tables fall back to frq).
- Main editor = piano-roll graph + numeric list, one phoneme at a time; file list has
  needs-review and unsaved columns.
- Selection: click one frame; drag contiguous range; Ctrl+click disjoint; Shift+click range
  between two; Ctrl+drag on graph (scope cursor) for range selection; Ctrl+A all.
- Editing: mouse paint/drag on graph; double-click the white key column area to set the whole
  table to one pitch (also paints unvoiced frames); numeric input + `<-set` button to overwrite
  selection; `-` sets selection to unvoiced; `x3`/`x2`/`/2`/`/3` multiply/divide the selection;
  `avg` linearly interpolates between the two ends of the selection (both ends unvoiced -> all
  unvoiced; one end unvoiced -> treated as ~56 Hz).
- **Buttons require double-click** (misclick protection).
- Value limits: 56 Hz .. 7902.13282010 Hz (A#1..B8); below -> unvoiced; above -> clamped to max.
- Key frequency: `平均値をキーに` button (mean of selection; select a stable region), direct numeric
  entry + Update, or drag the red key line on the graph. VS4U/w4u have no key in file -> cannot
  edit.
- Undo: menu undo; unsaved edits per file are retained across selection changes; close warning
  when unsaved.
- Playback: editor can play the (resampled) result while editing (3-17).

## 4. Auto check ("周波数表をチェック", 5-7)

Inputs: the table itself (pitch + volume), a single **recording key** for the folder, and three
options (all default off): `some pitch variation`, `big pitch change at start`, `big pitch change
at end`. Output: per-phoneme needs-review mark, red background ranges and candidate repair lines.

Processing order (5-7-1):

1. **Sound-group analysis** (5-7-4/5/6): using volume, find contiguous regions at/above a
   "voice present" level; each is one sound group. Pitch-less but loud frames are included.
   Isolated low-volume frames with a pitch line that does not connect to a group are excluded
   (likely noise/unvoiced). Store group start/end; all later checks never cross groups. The
   threshold can be absolute, histogram-based (recommended: classify silence/voice/boundary,
   then histograms), or a hybrid; exact value is left to the implementation.
2. **Unvoiced check** (5-7-7): inside a group, consider only frames between the first and last
   voiced frame. Unvoiced frames in that interval = broken. Before/after are excluded (likely
   unvoiced consonants / breath / edge phonation).
3. **Half/double pitch check** (5-7-8..5-7-16), run on the *unvoiced-repaired candidate* data,
   all voiced frames in each group:
   - **Absolute pitch check**: compare with the recording key. Without `some pitch variation`:
     >= 5 keys above -> double-pitch broken; >= 5 keys below -> half-pitch broken. With the
     option: >= 10 keys above/below.
   - **Relative pitch check** (conservative, adjacent frames only): walk the group; a change of
     >= 5 keys versus the previous frame marks a **break start** or **break end** — whichever
     side (current vs previous frame) is closer to the recording key is the normal side; the
     farther side is the start of a break. State machine:
     - end -> start: not broken.
     - start -> end: the interval is broken (start inclusive, end exclusive).
     - start -> start, end -> end: ambiguous, not broken; the second marker is discarded.
     - A leading end with no earlier start: group start .. (end - 1) is broken.
     - A trailing start with no later end: start .. group end is broken.
   - Broken if either absolute or relative says so.
   - Post-filter (5-7-16): with `big pitch change at start`/`end` on, if the first/last pitch of
     the group is inside a broken range, clear that range up to the next non-broken frame.
   - Known blind spots (documented as accepted): boundary frames with intermediate pitches may
     split a break into <5-key steps and be missed, and slow pitch glides across the recording
     key can cause false negatives/positives. Do not "fix" these with multi-frame comparisons
     (risk of new false positives).
4. **Repair candidates**:
   - Unvoiced repair (5-8-3), per broken span, applied first:
     - whole group broken -> fill with the recording key.
     - at group start -> copy the pitch of the first voiced frame after the span.
     - at group end -> copy the pitch of the last voiced frame before the span.
     - in the middle -> frqeditor `avg` (linear interpolation between surrounding pitches).
   - Half/double repair (5-8-4/5), per broken range, frame by frame:
     - reference pitch: whole group -> recording key; at start -> next clean frame; at end ->
       previous clean frame; middle -> the closer to the recording key of the two surrounding
       clean frames.
     - ratio = reference / frame_f0; snap to the nearest of `{1.5, 2, 3, 4, 2/3, 1/2, 1/3, 1/4}`,
       apply that multiplier to the frame. Per-frame application preserves pitch variation inside
       the range.
   - After repair, recompute key frequency = mean of voiced frames (5-8-2/5).
5. UI semantics: `fix` overwrites the selected needs-review span with the candidate, `ignore`
   clears it; edits are not auto-saved; undo works but the needs-review state is not restored;
   re-running check mainly returns to normal for repaired spans. `fix all` exists but is
   discouraged as error-prone. (5-4)

Scope limits (5-6, 5-9): designed for single/CV/VCV normal phonemes and voice-positive special
phonation; expects trouble with vibrato/melody/slur/edge phonemes (works with options) and
explicitly does not support death-voice, whisper, unvoiced-consonant-only, breath phonemes,
out-of-engine-range pitches, or large melodic jumps. It does not use the wav/spectrum (except
volume already in the table); improving with waveform-derived evidence is future work. Key
frequency improvements (median / weighted mean for melody banks) are out of scope in frqeditor.

## 5. Error-prone cases documented by the manual

- Low voice level in the sample vs noise: check misses breaks or flags noise (5-5-9/10).
- Half/double repair near the 5/10-key boundary can produce x1.5 candidates where a 2x/0.5x edit
  is needed (5-5-4); manual touch-up is expected.
- No-frequency decisions depend on phoneme policy: some producers set unvoiced consonants to the
  key frequency, others to `----`. Check options and repair must respect the chosen policy
  (5-5-7/8).

## 6. Feature opportunities vs frqeditor (informal, to be refined in issues)

- Per-frame waveform + spectrogram pane (frqeditor only has a spectrum view in VocalShifter
  workflow; the manual explicitly calls audio-based checking future work).
- Non-destructive undo history, session projects, multi-file batch compare.
- Better repair heuristics than 5/10-key thresholds; configurable ratios.
- Explicit support for `pmk`/`mrq` editing (frqeditor treats them as second-class).
- Scriptable CLI (batch generate + check + report), which frqeditor lacks entirely.
