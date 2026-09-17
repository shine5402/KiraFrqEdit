# Frame grid and count conventions (`frq`, `pmk`, `mrq`)

Status: **verified** against the local corpus (4,655 `.frq`, 1,231 `.pmk`, 50 `desc.mrq` with
1,981 entries, paired to ~4.7k wavs) and **reproduced black-box** by running the local engine
binaries (`resampler.exe`, `phavoco.exe`, `fresamp.exe`, `TIPS.exe`, `moresampler.exe`) on
synthetic wavs whose exact f0 and length are known. Answers the frame-convention question in
issue #2: this is the convention the KiraFrqGen writers should follow.

## Recommended rules

| Format | Entry positions | Entry count for a wav of `L` samples | Unvoiced |
| --- | --- | --- | --- |
| `frq` | frame `i` = f0/volume of window `[i*256, (i+1)*256)`, stored at `t = i*256/44100` | `N = floor(L/256) + 1` | f0 `0.0` |
| `pmk` | period segments: `pos_end` strictly increasing, advancing by one local period | number of period marks `< L` (≈ `L / mean_period`) | `code = 49`, step `49` |
| `mrq` | point `j` = f0 of window `[j*nhop, (j+1)*nhop)`, stored at `t = j*nhop/fs` | `nf0 = floor(L/nhop)` | f0 `<= 0` |

All corpus and probe wavs used here are 44.1 kHz. `frq` has no sample-rate field (44.1 kHz is
assumed); `mrq` carries `fs`/`nhop` and was `44100`/`256` in all 1,979 paired entries.

## `frq`

### Frame grid

Frame `i` is labelled at sample `i*256` (`t = i*256/44100`); its analysis window starts at that
sample, i.e. `[i*256, (i+1)*256)`:

- Black-box `resampler.exe` on a synthetic tone whose f0 steps at sample 22050: frame 86
  (sample 22016 = window start) already contains the step (227.30 Hz, ≈ 34 old + 222 new samples),
  frames ≤85 hold the old f0 and frames ≥87 the new one. A window ending at `i*256` would have put
  the step one frame later.
- A tone starting at sample 0 is voiced in frame 0, so there is no leading padding/offset.
- Consumer confirmation: OpenUtau `Classic/Frq.cs` converts ms to a frame index as
  `floor(ms * 44100 / 1000 / hopSize)`; the `hop` field is read from the file (256 everywhere).

### Frame count

`N = floor(L/256) + 1` (frames 0…`floor(L/256)`):

- Corpus: 4,513 paired files, `N - floor(L/256)` is **+1 for 4,089 (90.6 %)**, **0 for 418
  (9.1 %, all from one bank)**, and +2/+7/+12/+35/+41 for 6 stale tables (from two banks,
  where the wav was shortened after the table was generated, so `N` covers more samples
  than the wav now has). The 2-D matrix of `L mod 256` × `N - floor(L/256)` is flat: the rule does
  not depend on the remainder, so it is `floor(L/hop)+1`, not `round`/`ceil`.
- Black-box: `resampler.exe` produces `floor+1` at `L` = 100, 300, 44,100, 66,150, 82,500,
  87,040 (exactly 340×256) and 88,200; `phavoco.exe` agrees at `L` = 87,040; `fresamp.exe`
  produces `floor+2` at all successful probe lengths (one extra trailing zero frame).
- The last frame's window may run past the end (it always does when `L` is a multiple of 256);
  UTAU's resampler writes `0.0` there on synthetic probes, and real tables commonly end unvoiced.
- 142 further `.frq` in the corpus have no wav at all: stale copies in the orphan bank with
  mojibake names and ` (2)`/` (3)` duplicate suffixes.
- All-unvoiced tables still use `floor+1` (18 corpus files; probe silence `L=44,100` → `N=173`,
  all `0.0`).

### Other writers, for reference

`fresamp.exe` = `floor(L/256)+2`; one bank's writer (418 files, reserved bytes zero)
= `floor(L/256)`. No locally installed engine produced `floor+0`, so that bank was likely generated
by SpeedWagon or another tool; KiraFrqGen should use the UTAU-default `floor+1`.

## `pmk`

### Entries are period segments, not a hop grid

Every entry is `(pos_end, code)` where `code = round(44100/f0)` (49 = unvoiced) and `pos_end` is
the sample where that period/segment ends. `pos_end` is strictly increasing in all 1,231 files, and
the walk advances by one period at a time, not by 256: a 2 s pure tone at period 165 yields ~533
entries, a 4.56 s voice sample `aR_wav.pmk` has 2,350 entries (average step 85.6 samples because
most of it is unvoiced 49s).

- Start: `pos_end[0] = code[0]` in 1,214/1,231 files (98.6 %); for an unvoiced start `code = 49`,
  so the walk simply starts at sample 0. The 17 exceptions (12 in one VCV bank, 5 in another)
  are voiced-start files where the first mark is the first detected pitch mark (69…357), as TIPS
  also does black-box (first mark 287 on a tone that starts at 0).
- End: `pos_end` is always `< L` (1,230/1,231; the single exception pairs with a
  shortened stale wav, and its `.frq` is stale too). Rule: append marks while the *rounded* mark is
  below `L` — a mark that would round to `L` is suppressed, not clamped, so the final gap is
  `1…49` samples. Only 2 files were left with room for another
  whole period (both edge/stale cases: one with 2,555 samples to spare and
  one off by 15).
  - Black-box silence (`L=44,100`, step 49): entries end at 44,051 = 899×49; 900×49 = L exactly and
    is not emitted. `L=100` → 2 entries, `L=300` → 6 entries.
  - Black-box fractional phase (300 Hz tone, fixed silence tail, `L` stepped one sample over 120
    values): the final gap cycles 1…49 and never reaches 0, and a gap of 49 occurs (a clamp to
    `L-1` would cap the gap at 48); the corpus agrees — 0/1,231 files store `pos_end == L`, and
    gap-1 closers occur at the ~1/49 rate of a uniform gap rather than the ~50 % a clamp would give.
  - A wav that ends voiced stops at the last analysed window, a few hundred samples early
    (pure tone: `L=82,500` → last 82,292; `L=88,200` → last 88,067).

### `pos_end` generation / resync rule

There is **no resync at code changes**: `pos_end` continues as a single error-diffused accumulator
of TIPS's *unrounded* period estimate, while `code` is a separately quantised value. Writing
`pos_end[i] = pos_end[i-1] + code[i]` reproduces 84.5 % of transitions; the rest carry the
fractional part of the period.

Measured over 2,357,044 transitions in 1,231 files:

| transition kind | count | share |
| --- | --- | --- |
| `d == code[i]` | 1,991,017 | 84.47 % |
| `d == code[i-1]` and code changed | 64,490 | 2.74 % |
| other | 301,537 | 12.79 % |

Of the 301,537 "other" transitions, 207,569 deviate by exactly ±1 (+1: 111,292, −1: 96,277) —
error diffusion inside constant-code runs. Larger deviations concentrate at f0 transitions
(92,825 of them change code); the first voiced marks after an unvoiced run can deviate by roughly
half a period while the estimator settles (e.g. `aR_wav.pmk` 22,932 → 23,014 → 23,088 → 23,248…).

Black-box proof that code and walk are separate estimates: a pure 300.00 Hz tone (true period
147.000) is stored with `code = 146` in 99.2 % of entries (`avg` field 146.008) while the walk
mean step is 147.00. A sweep of pure tones at integer periods P = 140…180 gives walk step = P
(±0.01) everywhere, but stored `code = P−1` for P ∈ 145…148 and 167…175, `P+1` for P ∈ 155…160 —
the stored code never matches the walk exactly in those bands. The pitch/note argument
(C4/D4/A4/C5/MIDI 60/69/81) does not change the output.

**Consequence:** byte-exact generation of a TIPS-identical `.pmk` from a wav requires TIPS's own
estimator and is out of scope for a clean-room writer. A *compatible* encoder from a 256-grid f0
contour is unambiguous, though:

```text
pos = 0.0                                  # float sample position
repeat:
    f = f0 at `pos` (frame lookup, optional interpolation)   # 0/absent = unvoiced
    p = 49.0 if unvoiced else 44100 / f    # unrounded period
    if pos + p >= L: stop
    pos += p
    emit (round(pos), 49 if unvoiced else round(44100 / f))
```

- Use the unrounded period for the walk (matches TIPS's ±1 error diffusion); using the integer
  `code` for both is simpler and stays inside the observed deviation envelope, but it can end the
  table up to one period early/late.
- Never re-quantise `pos_end` on load or round-trip: preserve stored `(pos_end, code)` verbatim
  (same policy as `pmk-format.md`).
- `code = 49` for unvoiced and exactly 49-sample steps there (4 corpus files are all-49; the
  silence probe walks 49, 98, …, 44,051).

## `mrq`

Point `j` is the f0 of window `[j*nhop, (j+1)*nhop)` at `t = j*nhop/fs`, and the count is exactly
the number of *complete* windows:

`nf0 = floor(L/nhop)`

- Corpus: 1,979 entries paired to wavs, `nf0 = floor(L/nhop)` for **1,974 (99.7 %)**; the 5
  exceptions are stale caches in one bank where the cached analysis is longer than the
  current wav (`nf0 > floor(L/nhop)`, by 512 up to ~35k samples — the same wav-shortening seen in
  that bank's `.frq` tables). No entry used `floor+1`, `ceil` or `round` (`round` "matches" 851
  only by coincidence when `L mod nhop >= nhop/2`).
- Black-box `moresampler.exe`: `nf0` = 344 (`L=88,200`), 340 (`L=87,040`), 258 (`L=66,150`).
  A synthetic f0 glide 200→300 Hz over 2 s fits `f0[j] = 200.000 + 50*(j*256/44100)` — grid origin
  0 (fit error 0.2 samples), confirming the `t = j*nhop/fs` labelling. The first 1–2 points and
  some last points are `0.0` while the LLSM estimator warms up.
- Because `nf0 = floor(L/nhop)`, no point represents the trailing partial window; the last point is
  at `(nf0-1)*nhop ≤ L - nhop` (trailing residuals in the corpus are all in `[256, 512)`).
- Deleted entries carry a zero-length name and are skipped via `size`; two entries in one bank have a
  `size` larger than `20 + 4*nf0` (e.g. 308 vs 284), so trust the `size` field, not the formula.

## Pairing wavs to sidecars

- `frq`/`pmk`: sidecar `<name>_wav.frq|.pmk` sits next to the wav and corresponds to either
  `<name>_wav.wav` (engine temp convention) or `<name>.wav` (bank source file). Corpus examples:
  `aR_wav.pmk` ↔ `aR.wav`, `aR_wav.frq` ↔ `aR.wav`. All 1,231 `.pmk` and 4,603 `.frq` use
  `_wav.`; `.wav.frq` (a documented UTAU/OpenUtau name) occurs 0 times locally, but accepting both
  is required by `frq-format.md`'s lookup rule.
- `mrq`: one `desc.mrq` per folder, entry name = the bare wav filename (`UTF-16LE`, no path).
  Matching is a linear filename compare in the same folder, independent of `oto.ini`.
- **Mojibake warning:** 1,248/1,981 entry names in the corpus were written as CP932→CP936
  mojibake (`_偄傫偄偄…` instead of `_いんいい…`) because moresampler ran under a Chinese locale.
  Recover with `name.encode("cp936").decode("cp932")` before pairing; otherwise 1,250 entries
  appear to have no wav. Two entries in one bank have all-NUL names (skip them).

## Edge cases

| Case | `frq` | `pmk` | `mrq` |
| --- | --- | --- | --- |
| `L` < one hop (100/300) | `N = floor+1` (1/2 frames), all `0.0` | 2/6 entries, all 49s | moresampler crashes, writes no entry; `desc.mrq` header only |
| all-unvoiced wav | `floor+1` frames, all `0.0` | exact 49-sample walk | not produced by moresampler (crash on pure silence) |
| trailing partial frame | included as the last frame (`floor+1`), usually unvoiced | not emitted (walk stops before `L`) | not represented (`floor(L/nhop)` only) |

`fresamp.exe` crashes on silence/short wavs, writing a 0-byte `.frq`.

## Sources

- Local corpus (read-only; path in `AGENTS.local.md`), 4,655 `.frq`, 1,231 `.pmk`,
  50 `desc.mrq` (1,981 entries), all 44.1 kHz.
- Black-box runs of the clean engine builds in `engines_original/`
  (`resampler.exe`, `phavoco.exe`, `fresamp.exe`, `TIPS.exe`, `moresampler.exe`); binaries are
  proprietary and were only observed, never inspected.
- OpenUtau [`Classic/Frq.cs`](https://github.com/openutau/OpenUtau/blob/master/OpenUtau.Core/Classic/Frq.cs)
  (ms→frame mapping, hop from header),
  [`worldline/classic/frq.cpp`](https://github.com/openutau/OpenUtau/blob/master/cpp/worldline/classic/frq.cpp)
  (layout).
- `docs/research/frq-format.md`, `pmk-format.md`, `mrq-format.md` (layout and unvoiced semantics).
- frqeditor manual (`manual.extract.txt`): TIPS generates `pmk` on demand (2-3-4), SpeedWagon
  generates `frq` (2-3-3), 256-sample volume convention (4-1-5), VS4U2PMK sample source (3-9-3).

## Local verification

Scripts (worktree `.local/`, gitignored; corpus read-only):

- `frame_scan.py`, `frq_grid.py`, `frq_fingerprint.py` — `frq` pairing/count matrix/per-bank.
- `pmk_rules.py`, `pmk_probe.py` — `pmk` end rule, first entry, 2.36 M transition statistics.
- `pmk_end_scan.py`, `pmk_end_probe.py` — final `pos_end` gap: corpus scan and the `TIPS.exe`
  length sweep behind the end rule above.
- `mrq_pair2.py` — `mrq` pairing incl. mojibake repair, `nf0` formula counts.
- `engine_probe.py`, `probe_analyze.py`, `engine_matrix.py`, `boundary_probe.py`,
  `align_probe.py`, `pmk_sweep.py`, `pmk_pitch_probe.py`, `final_checks.py` — synthetic wavs and
  black-box engine comparisons.
