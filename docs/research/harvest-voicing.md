# Stopping Harvest from voicing noise: levers on the lightweight WORLD path

Status: research for issue [#50] (2026-09-19). Reads the vendored WORLD v1.0.1 source
(`third_party/World`, commit `d625e76`), the current shim/wrapper, and upstream WORLD issues.
Line numbers are for the **vendored** `src/harvest.cpp`, which carries the #34 progress patch
(`third_party/World/patches/0001-progress.patch`), so they run a few lines ahead of pristine
upstream. The levers are ranked, not implemented, except lever 2 — the D4C aperiodicity gate —
which #55 decided to adopt and #64 landed (see that section).

[#50]: https://github.com/shine5402/KiraFrqEdit/issues/50

## TL;DR

- Harvest over-voices **by design**. Its author states it "attempts to reduce the unvoiced frame
  and give it a reliable F0" and that the voiced/unvoiced decision is meant to come from **D4C
  aperiodicity, not from `f0 == 0`** ([mmorise/World#105], [mmorise/World#35],
  [mmorise/World#21]). UTAU `frq` has no aperiodicity field — its semantics are `f0 == 0` ⇒
  unvoiced — so if we want fewer voiced frames we must gate Harvest's `f0` ourselves.
- The "~0.05 threshold" named in issue [#50] is `RemoveUnreliableCandidatesSub`'s
  neighbour-continuity prune (`harvest.cpp:663`). It is **not** the main voiced/unvoiced gate,
  and changing it alone is unlikely to fix over-voicing: `FixStep3` (extension) and `FixStep4`
  (gap filling) can re-add voiced frames afterwards.
- The post-processing stages dominate the final voiced set: `FixStep2` drops voiced runs
  shorter than 6 frames (`:1043`), while `FixStep3` extends voiced sections up to 100 frames
  (`:871`, `:800`) and `FixStep4` fills unvoiced gaps shorter than 9 frames (`:1046`). The last
  two can re-add exactly the frames the confidence gates removed.
- Cheapest lever: **use DIO** (it voiced fewer frames than Harvest in our corpus test and is
  already exposed). Best Harvest-specific payoff on principled grounds: a **D4C aperiodicity
  gate** — the mechanism Harvest was designed to be paired with. Best cost/risk when patching:
  **expose the duration knobs** through `HarvestOption`.
- No upstream WORLD fix exists: `harvest.cpp` is unchanged since 2021-02-15 (v1.0.1); the
  over-voicing is intentional, not a bug that will be fixed.

## How Harvest decides voiced vs unvoiced

Harvest estimates on a 1 ms internal grid then resamples to `frame_period`
(`HarvestGeneralBody`, `harvest.cpp:1151`; the 1 ms→`frame_period` copy is
`Harvest`, `:1229-1261`). A frame's `f0` ends at exactly `0.0` only when it is zeroed by one of
these steps, in order:

| # | Step | Vendored location | Default | What it does |
| --- | --- | --- | --- | --- |
| 1 | Channel candidate accepted | `GetF0CandidateContourSub` | ±10% of channel f0, within `f0_floor..f0_ceil` | Keeps a channel's f0 only if it tracks that channel and is in range (`:246-247`, `:253-254`) |
| 2 | Channel section dropped | `DetectOfficialF0CandidatesSub2` | `ed[i] - st[i] < 10` | Ignores channel runs shorter than 10 channels (`:377`) |
| 3 | Harmonicity / confidence gate | `GetRefinedF0` | `*refined_score < 2.5` | Zeroes a candidate whose instantaneous-frequency score is below 2.5, or out of `f0_floor..f0_ceil` (`:616-620`) |
| 4 | Neighbour-continuity prune | `RemoveUnreliableCandidatesSub` | `threshold = 0.05` | Zeroes a candidate unless some candidate in the previous **or** next frame is within 5% relative f0 (`:663`, `:665-672`) |
| 5 | Rapid-jump removal | `FixStep1` | `allowed_range = 0.008` | Zeroes frames that jump >0.8% from the extrapolated contour and >0.8% from the previous frame (`:716-728`, call `:1042`) |
| 6 | Short-run removal | `FixStep2` | `voice_range_minimum = 6` | Zeroes voiced sections shorter than 6 frames (`:754-768`, call `:1043`) |
| 7 | **Voiced-section extension** | `FixStep3`/`Extend`/`ExtendF0`/`ExtendSub` | `100` frames, `4` misses, `2200`, `allowed_range = 0.18` | Walks up to 100 frames outward from each voiced edge, adopting candidates within 18%, stopping after 4 consecutive misses; keeps only sections longer than `2200/mean_f0` frames (`:797-884`, call `:1045`) |
| 8 | **Short unvoiced-gap filling** | `FixStep4` | `threshold = 9` | Linearly interpolates over unvoiced gaps shorter than 9 frames, i.e. voices them (`:1006-1028`, call `:1046`) |

After step 8, `SmoothF0Contour` (`:1085-1119`) low-passes each voiced section
(zero-lag Butterworth, coefficients `:1087-1090`); it only writes voiced regions, so it does not
change voicing.

Key consequence: steps 3–6 are the pruning (anti-voicing) stages; steps 7 and 8 then
deliberately re-add voiced frames, which is exactly the behaviour the author describes.
Effective search range is also widened to `0.9 * floor` / `1.1 * ceil`
(`adjusted_f0_floor`/`adjusted_f0_ceil`, `:1155-1156`), and the internal
`channels_in_octave = 40` (`:1234`) and `overlap_parameter = 7` (`:1185`) are hard-coded.

### What the ~0.05 threshold actually is

`RemoveUnreliableCandidatesSub` (`:658-673`) takes each surviving candidate, asks
`SelectBestF0` for the closest candidate in frame `i+1` and `i-1` (within 100%, so any), takes the
smaller relative error `min_error`, and returns (keeps the candidate) only if
`min_error <= 0.05` (`:663`, `:670-672`). So it is a **temporal-continuity** prune on candidates,
upstream of the contour-level post-processing. Lowering it makes candidates stricter; raising it
makes them laxer. Because steps 7–8 rebuild the contour from whatever candidates survive and can
bridge gaps, this knob is indirect for our purpose.

### Why Harvest ends up more voiced than DIO

The author's stated intent is the opposite of what a UTAU table wants:

> "Harvest attempts to reduce the unvoiced frame and give it a reliable F0, so this result is
> reasonable as expected." — mmorise, [mmorise/World#105]

> "Harvest was designed to reduce the error that the voiced section is wrongly identified as the
> unvoiced section. ... The fricative therefore tends to be counted as the voiced section in cases
> where Harvest is used. D4C LoveTrain extracts the fricative identified as the voiced section and
> give it the 1.0 of aperiodicity." — mmorise, [mmorise/World#21]

That last quote is the officially-sanctioned answer to our problem: **use D4C aperiodicity as the
VUV decision**. The author also recommends DIO + StoneMask when periodicity is low
([mmorise/World#83]) and declined to expose `allowed_range` for Harvest because its many internal
parameters are jointly tuned ([mmorise/World#145]):

> "Harvest has many parameters to obtain an accurate F0 estimation, and all parameters have been
> optimized using a speech database. Their balance is important, and changing only the
> allowed_range may decrease the performance."

## What we already measured (do not re-litigate)

From `docs/research/world-integration.md`, on one 5.48 s 44.1 kHz mono corpus sample, at floor/ceil
71/800 Hz, `frame_period = 5 ms`: **DIO voiced 679/1097 frames, Harvest 719/1097** (mean f0 within
0.2 Hz of each other). Thus Harvest voiced ~40 more frames (~3.6% of the file) on that sample.
StoneMask does not change voicing. A pure single sinusoid makes Harvest voice almost nothing
(1/601) while DIO voices all — a synthetic-fixture gotcha, not a real-voice result. Treat the
40-frame figure as one sample, not a distribution; no one has yet measured voiced-frame counts
across the corpus for a DIO-vs-Harvest gap.

## Ranked levers

Ranked by likely payoff per unit cost/risk, for "stop voicing noise while staying on WORLD
(no ML)". "Cost" is implementation effort in this repo; "risk" is regression to real voiced
frames, since over-gating is as harmful as under-gating for a singer's `frq`.

| # | Lever | Payoff | Cost | Risk | Where it lands |
| --- | --- | --- | --- | --- | --- |
| 1 | **Use DIO, or DIO as a voicing mask on Harvest** | Medium; immediate | None (already exposed) / Low (mask in Rust) | Low–medium (DIO drops soft voiced at low SNR) | already reachable; `kirafrq-world-binding` + pipeline |
| 2 | **D4C aperiodicity gate on Harvest f0** | High (targets fricatives by design) | Medium–high (build `d4c.cpp`, shim fn, Rust API, threshold) | Medium (extra pass; threshold choice) | new shim call + post-process |
| 3 | **Expose the duration constants via `HarvestOption`** | Medium–high (directly removes extension/fill) | Medium (vendored patch + shim args) | Medium (joint tuning; per-file voicing looks different) | `harvest.h`/`harvest.cpp` patch + shim + Rust |
| 4 | **Energy/RMS voicing gate in post** | Low–medium | Low (Rust only, or reuse frq amplitude) | Medium (cuts soft onsets/phonation) | pipeline / writer |
| 5 | **Floor/ceiling tightening** | Low (only if noise sits outside the voice band) | None (already exposed) | Low | `F0Options` |
| 6 | **Pre-estimation high-pass / noise gate** | Low (Harvest already removes per-frame DC) | Low–medium | Medium (gate cuts breathy onsets; fricatives are not low-level) | pipeline before `estimate_f0` |
| 7 | **Tune `RemoveUnreliableCandidatesSub` 0.05 / score 2.5** | Low alone (re-add steps dominate) | Low (patch) | Medium–high (author's joint-tuning warning) | `harvest.cpp` patch |

### 1. DIO, or DIO as a voicing mask on Harvest

DIO exposes the knobs Harvest hides: `channels_in_octave` (default 2.0), `speed`, and
`allowed_range` (default 0.1; `dio.h:16-23`, `dio.cpp:655-670`). `allowed_range` is DIO's
jump-elimination threshold (`dio.cpp:137-155`), the analogue of Harvest's hard-coded
`0.008`/`0.18`. In the ecosystem, UTAU-facing WORLD tools mostly choose DIO for `frq`
generation: SpaceWorld's `writeUTAUfrq` calls `DioWithFramePeriod` + `StoneMask` (read from its
GPL `SpaceWorld/src/main.cpp`), and OpenUtau's `Frq.Build` accepts whatever `f0` an engine gives
it with no gate (`OpenUtau.Core/Classic/Frq.cs`). `frq0003gen` uses pYIN or DIO, not Harvest.

Two cheap variants, no patching:

- **Default to DIO** for `frq` generation. Measured to voice fewer frames here, 10x faster, and
  matches other UTAU tools. The author himself recommends DIO when periodicity is low
  ([mmorise/World#83]).
- **DIO-as-mask**: run DIO and Harvest, keep Harvest's `f0` only where DIO also voiced. This
  uses DIO as the VAD while keeping Harvest's pitch quality. Cost is one extra (cheap) pass.
  Risk: DIO's own voiced→unvoiced errors become ours; and on the one sample the gap was only
  ~3.6%.

Recommendation: make this the first experiment — it is free and measurable.

### 2. D4C aperiodicity gate (the principled Harvest fix)

**Adopted (#55 decision, #64 build), as a supplement to the energy gate.** On the tuned
(`WORLD quirks`) path, Harvest only: force unvoiced where the raw `D4CLoveTrain` ratio is below
**0.85** (D4C's own default split, an internal constant with no user knob), applied after the
energy gate. Measured pooled over 6 banks / 240 wavs against RMVPE (conf ≥ 0.03): energy alone
removed 75.0% of the raw spurious frames at a 2.51% raw true-voicing cut; energy + D4C reached
86.0% at 4.29%, clearing 43.8% of the energy gate's residual at a 1.01% incremental cut. D4C
alone is no replacement (18.3% at 1.19%). Cost is ~2.5–3.6% of Harvest+StoneMask. DIO is
unchanged. The gate only ever forces frames unvoiced; a D4C computation failure is a per-file
failure like StoneMask, and a silence-only / no-voiced file is a no-op. The mechanism below is
the research that led to it.

`d4c.cpp` (compiled since #63) has `D4CLoveTrain`, which computes a per-frame band-energy ratio
`aperiodicity0` (`d4c.cpp:227-285`); `D4C()` skips aperiodicity estimation for frames where
`f0[i] == 0 || aperiodicity0[i] <= option->threshold` (`d4c.cpp:385-386`), default threshold
`0.85` (`constantnumbers.h:37`, `d4c.cpp:405-406`). `D4CLoveTrain` was added specifically to
re-label fricatives that Harvest called voiced ([mmorise/World#21]). Its ratio is not itself the
output aperiodicity: in the final `D4C` aperiodicity, the 0 Hz bin lands near 1.0 for unvoiced
frames and near 0.001 for voiced frames ([mmorise/World#35]).

The gate runs `Harvest` (post-StoneMask) → the `D4CLoveTrain` statistic → set `f0[i] = 0` where
the statistic is below the threshold. One subtlety: the `aperiodicity0` computed by
`D4CLoveTrain` (`d4c.cpp:249`) is a **periodicity** ratio (high = voiced), not the final
aperiodicity. D4C's own rule is `f0[i] == 0 || aperiodicity0[i] <= threshold` ⇒ unvoiced
(`d4c.cpp:386`), with `threshold = 0.85`; "if the Love Train estimate is below this threshold,
aperiodicity 1.0 is used (same as for unvoiced frames)" ([mmorise/World#21]). So the gate forces
unvoiced where **`aperiodicity0 < T`** (T = 0.85). The alternative — the final
`aperiodicity[i][0]` from `D4C()`, which lands at ≈1.0 for unvoiced and ≈0.001 for voiced
(`InitializeAperiodicity` default `1.0 - eps`, `d4c.cpp:323-328`; `coarse_aperiodicity[0] = -60.0`
⇒ 0.001, `d4c.cpp:373`), with the author's suggested split at 0.5 ([mmorise/World#35]) — was not
taken. This is the only lever that targets the actual failure class (breathy consonants) rather
than overall energy, and it stays inside WORLD (no ML). The vendored
`0002-d4c-aperiodicity0.patch` exposes the statistic as `GetD4CAperiodicity0()`, surfaced as
`kirafrq-world-binding::d4c_aperiodicity0()` and gated in `kirafrqgen-core`.

### 3. Expose the duration constants through `HarvestOption`

Today `HarvestOption` has exactly three fields (`harvest.h:16-20`) and the shim sets only those
three (`world_shim.cpp:47-58`), mirrored by `F0Options` (`crates/kirafrq-world-binding/src/lib.rs:9-24`)
and the core seam (`crates/kirafrqgen-core/src/lib.rs:68-87`). A minimal, upstream-friendly patch
adds fields with defaults equal to today's constants so unused callers are byte-identical:

```c
typedef struct {
  double f0_floor;
  double f0_ceil;
  double frame_period;
  int    voice_range_minimum;   // FixStep2, default 6
  int    max_unvoiced_gap;      // FixStep4, default 9
  int    max_extension;         // Extend,   default 100
} HarvestOption;
```

`InitializeHarvestOption` sets the new defaults; `FixF0Contour`/`HarvestGeneralBody` read them
instead of the literals at `:1043`, `:1046`, `:871`. This is the highest-payoff *Harvest-specific*
lever that does not require adding D4C, because it lets a caller clamp the two behaviours that
cause over-voicing (extension and gap filling) while leaving the jointly-tuned confidence
thresholds alone — directly answering the author's objection in [mmorise/World#145]. Exposing
`allowed_range` too would let a caller match DIO's tunable behaviour, but is riskier.

Caveat: `max_unvoiced_gap` and `max_extension` bound durations in **1 ms internal frames**, not
output frames; document that. Also, changing them changes the contour shape for genuine short
gaps, so this wants a corpus comparison, not a default flip.

### 4. Energy/RMS voicing gate in post

Cheap and patch-free: compute per-frame RMS (or reuse the `frq` amplitude the writer already
computes) and zero `f0` where the frame is below a level threshold. This is a generic VAD and
does not know about harmonicity. It helps against silence and low-level background, but breathy
consonants can be as loud as a soft vowel, and soft voiced onsets/phonation are exactly the
frames this risks cutting — the classic VAD trade-off. If used, prefer a relative (per-file
percentile) threshold over an absolute one, and leave a hysteresis.

### 5. Floor/ceiling

Already exposed and free (`F0Options`, `world_shim.cpp:34-58`). Raising `f0_floor` above the
pitch of low rumble, or lowering `f0_ceil` toward the singer's range, removes only frames whose
estimated pitch lies outside the band. It cannot help mid-band fricatives and risks dropping
genuine high passages; use as a guard, not a fix. Note the internal search band is still
`0.9*floor..1.1*ceil` (`harvest.cpp:1155-1156`).

### 6. Pre-estimation high-pass / noise gate

- **DC removal is already done** inside Harvest per frame (`GetWaveformAndSpectrum`,
  `harvest.cpp:84-89`), so a DC-removal pre-pass is redundant for the spectrum Harvest reasons
  about. A high-pass at ~50–80 Hz would additionally remove sub-band rumble/handling noise that
  the per-frame mean does not; low cost, low expected payoff, low risk.
- **A noise gate** before estimation is the weakest option: Harvest's decision is driven by
  zero-crossing consistency and harmonic score, not frame energy, so gating helps only when it
  removes the signal entirely. Meanwhile it attacks the very onsets (soft/breathy attack) that
  matter musically. Published voicing-aware front-ends (e.g. SRH's LPC pre-whitening and
  residual-harmonic voicing, Drugman & Alwan) improve voicing accuracy under additive noise by
  *spectral whitening*, not by gating — that is closer to what WORLD already does internally, so
  we do not expect a large win from generic DSP here.

### 7. Tuning the 0.05 / 2.5 thresholds

Patchable but indirect. Lowering `0.05` prunes more candidates before extension; raising `2.5`
rejects less-harmonic candidates. Both are the constants the author warned are jointly tuned
([mmorise/World#145]), and both sit upstream of the re-adding steps 7–8, so a change is hard to
predict without a corpus run. Keep as a last-mile tuning knob, not a first move.

## Upstream and other tools

- The over-voicing is documented intent, not a defect: [mmorise/World#35], [mmorise/World#105],
  [mmorise/World#21]. The recommended VUV source is D4C aperiodicity at 0 Hz (`> 0.5` ⇒
  unvoiced; mmorise in [mmorise/World#35]).
- There is no upstream fix to backport: `git log` for `src/harvest.cpp` on `mmorise/World` shows
  the last substantive change was 2021-02-15 ("Formal release version"), i.e. v1.0.1 is current;
  nothing since touches voicing.
- [mmorise/World#145] (why no `allowed_range` for Harvest) is the author declining the exact
  "expose a knob" request, for joint-tuning reasons. [mmorise/World#83] recommends switching to
  DIO + StoneMask when periodicity is low.
- [mmorise/World#151] is only an SPTK/diffsptk announcement; no new voicing API.
- Other WORLD-based UTAU tools read for behaviour (not code): SpaceWorld uses
  DIO + StoneMask + D4C and no explicit voicing gate for its `frq` writer; OpenUtau's
  `Frq.Build` takes `f0` as-is; `frq0003gen` uses pYIN/DIO. None of them use Harvest for `frq`.
- RVC's "Fix F0 predictor for Harvest" commit is about interpolating unvoiced frames for ML
  training, not about reducing voicing; not applicable.

## Recommendation

1. **Immediately measurable, zero patch: compare DIO, Harvest, and DIO-masked-Harvest** across
   the local corpus (voiced-frame counts and the voiced set, not just means). If DIO-masking
   removes the spurious frames without eating real ones, ship that; it is the cheapest fix and
   aligns with the rest of the UTAU ecosystem.
2. **The D4C aperiodicity gate — adopted (#55/#64).** It is the fix the WORLD author designed
   Harvest to be paired with, and it targets fricatives rather than energy; it landed as a
   supplement to the energy gate (see lever 2).
3. **If a patch is acceptable: expose the duration constants** (`voice_range_minimum`,
   `max_unvoiced_gap`, `max_extension`) on `HarvestOption` with current-value defaults; it is a
   small, documented patch that leaves the tuned confidence thresholds alone.
4. Treat pre-DSP (HPF/gate) and floor/ceiling as guards, not fixes.

## Uncertainty / open questions

- The one measured DIO-vs-Harvest voiced-frame gap (679 vs 719 of 1097) is a **single sample**
  from `world-integration.md`; no corpus-wide distribution exists yet, so lever 1's payoff is
  unquantified.
- ~~Whether to gate on the raw `D4CLoveTrain` ratio (default split 0.85) or the final
  `aperiodicity[0]` (author's split 0.5), and where to put the threshold, are unmeasured.~~
  Resolved by #55: the raw statistic at **0.85** is the knee and D4C's own default, and the gate
  ships Harvest-only on the tuned path (lever 2).
- The duration-constant patch changes the contour only in short gaps/runs; its effect on real
  voiced onsets (which is what `Extend`/`FixStep4` were added to protect) is exactly the risk and
  needs a corpus A/B.
- `f0` written to `frq` is later resampled from the 5 ms WORLD grid to the 256-sample UTAU grid
  (`f0-generation.md`); whether a voicing gate before or after that resample is safer is open.

## Sources

- Vendored WORLD v1.0.1 source: `third_party/World/src/harvest.cpp` (all line refs above),
  `harvest.h:16-20`, `d4c.cpp:227-285,385-386,405-406`, `world/constantnumbers.h:37`,
  `world/dio.h:16-23`, `dio.cpp:137-155,648-671`.
- Shim/wrapper: `crates/kirafrq-world-binding/shim/world_shim.cpp:47-58`,
  `crates/kirafrq-world-binding/src/lib.rs:9-24,89-135`, `build.rs:3-10`;
  core seam `crates/kirafrqgen-core/src/lib.rs:68-87`.
- `docs/research/world-integration.md` (measured DIO/Harvest voiced counts, pure-tone gotcha,
  StoneMask does not change voicing), `docs/research/f0-generation.md` (grid/resample plan).
- Upstream issues: <https://github.com/mmorise/World/issues/21> (D4C threshold / fricative
  intent), <https://github.com/mmorise/World/issues/35> (D4C AP at 0 Hz for VUV),
  <https://github.com/mmorise/World/issues/46> (DIO vs Harvest noise), <https://github.com/mmorise/World/issues/83>
  (use DIO when periodicity is low), <https://github.com/mmorise/World/issues/105> (Harvest
  voices silence, by design), <https://github.com/mmorise/World/issues/145> (no `allowed_range`
  for Harvest, intentionally), <https://github.com/mmorise/World/issues/151> (SPTK/diffsptk
  announcement). `git log -- src/harvest.cpp` on `mmorise/World` (last: 2021-02-15).
- Other tools (behaviour read only, not copied): OpenUtau `OpenUtau.Core/Classic/Frq.cs`
  (`Frq.Build`), SpaceWorld `SpaceWorld/src/main.cpp` (`writeUTAUfrq` = DIO + StoneMask; D4C is
  used elsewhere in its resampler), `titinko/frq0003gen` (pYIN/DIO).
- Voicing-under-noise literature: T. Drugman & A. Alwan, "Joint Robust Voicing Detection and
  Pitch Estimation Based on Residual Harmonics", arXiv:2001.00459 (LPC whitening + residual
  harmonics improve voicing decisions under noise).
