# `frq` amplitude scale: what writers store, and what KiraFrqGen should write

Status: **corpus-verified** against 4,512 paired `frq`+`wav` files in the local UTAU corpus, plus
one open-source writer (OpenUtau) and the frqeditor manual. One corpus writer
is named by its header marker (SpeedWagon); the others are distinguishable by their reserved bytes
and amplitude formula but cannot be named from the files alone. The recommended formula is at the
end. This resolves issue #3; the final policy call sits in #8.

## TL;DR

- Amplitude is display/relative data — no resampler is known to read it (`frq-format.md`).
  frqeditor **displays** it but cannot edit it (manual 3-13-17, 4-2-1), so a generated table is
  stuck with the values KiraFrqGen writes.
- Four exact per-frame formulas cover ~92% of the paired corpus. With `x` the wav samples and
  frame `i` = `[i*256, (i+1)*256)`:

  | Formula | files | writer evidence |
  | --- | ---: | --- |
  | `mean(|x|) * 2^15` | 683 | SpeedWagon marker (`" speedwagon     "`); also OpenUtau source |
  | `rms(frame) / sqrt(2) * 2^15` | 2,688 | unnamed, dominant; zero reserved bytes |
  | `rms(frame) * 2^15` | 420 | `55.0`-unvoiced writer (unvoiced stored as `55.0`, not `0.0`) |
  | `3 * rms([i*256, i*256+1024))` | 363 | unnamed; writes `44100` at 0x14 |
  | none of the above | 181 | the orphan bank (mangled-name orphan tables) + stragglers |

- **Recommendation:** write `amp[i] = 2^15 * mean(|x[j]|)` over the 256-sample frame — the
  OpenUtau `Frq.Build` formula, which also matches SpeedWagon's corpus output. Details below.

## Method

- Paired every `*_wav.frq` with `<stem>.wav` (4,601/4,654 local frq files are named `<name>_wav.frq`;
  4,512 had a wav; the 142 orphans are mojibake-named leftover tables in the orphan bank,
  which has 284 frq for 142 wavs).
- For each file: decoded the wav (all corpus wavs are 44.1 kHz mono 16-bit), computed per-frame
  `mean(|x|)`, `rms`, `peak`, and boxcar variants (windows 256–4,096, offsets -256..+256), then
  took the per-file median ratio `amp / candidate` and the fraction of frames within 1%/5% of it.
- Grouped files by reserved bytes at 0x14 and by bank.
- Scripts: `.local/{scan,correlate,classify,window_search,final_stats,pmk_frq_check,fine_hist}.py`
  (throwaway; regenerate the TSVs before re-running the later stages).

## Provenance markers (4,654 files)

| Reserved bytes at 0x14 | Files | Meaning |
| --- | ---: | --- |
| all zero | 3,397 | multiple writers (see below) |
| `" speedwagon     "` (ASCII) | 895 | **SpeedWagon** (Custom.Maid's dedicated frq generator) |
| `44 AC 00 00` = int32 `44100` | 362 | sample rate; writer unidentified |

This corrects `frq-format.md`: `44100` is stored in **362** files, not 1,253. "Non-zero reserved"
is 1,257; most of those (895) are the SpeedWagon marker.

Naming is not a discriminator: essentially every local frq is `<name>_wav.frq`. `44100` and SpeedWagon files
are interleaved within the same banks (e.g. one bank has 217 `44100` + 70 zero; another
has 22 `44100` + 1,929 zero), so banks were re-generated with different tools over time.

A second fingerprint: two banks write **`f0 = 55.0` for unvoiced frames** instead of `0.0` —
the `55.0`-unvoiced bank (420 files) and the orphan bank (284), plus 3 stragglers. Tools like
oatsu's utau_tools map
`<= 55` to unvoiced for exactly this reason; standard resamplers instead see a real ~55 Hz pitch.
Do not imitate this.

## Formula fits per group

Ratios are `amp / formula`, per-file medians; "within-file" is the median fraction of frames
within 5% of that file's median.

| Group | n | Formula | median ratio | p10–p90 | files <1% | files <5% | within-file |
| --- | ---: | --- | ---: | --- | ---: | ---: | ---: |
| SpeedWagon marker | 726 | `meanabs256 * 2^15` | 0.9950 | 0.983–1.005 | 67% | 99.6% | 32% |
| zero, dominant | 2,683 | `rms256/sqrt(2) * 2^15` | 1.0006 | 0.9967–1.0060 | 94.5% | 100% | 75% |
| zero, `55.0`-unvoiced | 425 | `rms256 * 2^15` | 1.00000 | (exact) | ~100% | ~100% | 100% |
| reserved `44100` | 357 | `3 * rms1024` | 0.9994 | 0.9969–1.0007 | 99.4% | 99.7% | 85% |
| zero, orphan bank | 142 | — (no candidate fits) | — | — | — | — | ~12% |

Notes:

- `rms256` is `sqrt(sum(x^2)/256)` over the frame; `rms256/sqrt(2)` is mathematically
  `sqrt(mean(x^2)/2)`, i.e. the writer divides the mean square by 2. The zero-reserved group is
  overwhelmingly one writer (2,688/3,397 files; e.g. one bank 1,929, another 502,
  two more 54 and 15, plus parts of three others).
- The `55.0`-unvoiced writer's fit is exact frame-by-frame (10 files re-checked: ratio `1.00000`,
  100% of frames within 1%) — different from the `/sqrt(2)` writer despite both being zero-reserved and
  both RMS-based.
- The `44100`-writer's 3x factor is exact (99.4% of files within 1% of 3.0) and its window is a
  full **1,024 samples anchored at the frame start**, not a smoothed 256-window. Why 3x is
  unknown; writer identity is unknown (see uncertainty). It is emphatically *not* a `pmk`
  conversion: only 55/362 have a `.pmk` sibling, and matching frq f0 against those pmk segments
  fails (`pmk_frq_check.py`).
- SpeedWagon's per-file scale is solidly `meanabs * 2^15`, but its envelope is slightly wider or
  smoothed than the plain 256-frame boxcar (per-frame spread only ~32% within 5%; a ~1,024-wide
  mean-abs window raises a sample file to ~83%). The scale is what matters here.
- The orphan bank's writer's amplitude tracks neither the wav's mean-abs nor its RMS under any tested
  window (e.g. one orphan file: amp rises 973 → 8,235 while frame RMS is flat ~2,500–4,000). The bank
  contains many mangled-name orphan frq files, so the paired wav is probably not the analysis
  source. Treat as unknown; not a model for KiraFrqGen.

Key-frequency style (secondary): SpeedWagon and `44100` match **median of voiced f0** (654/728 and
306/357 of the files checkable against a paired wav, within 2%); the zero group is mixed
(median/arithmetic/geometric); frqeditor's own repair rule is the arithmetic mean of voiced frames
(manual 5-8-5). Left to #8.

## What OpenUtau writes (primary source)

`OpenUtau.Core/Classic/Frq.cs`, `Frq.Build(float[] samples, double[] f0)`, condensed:

```text
ampMult = 2^15                              // 32768
amp[i] = sum(|samples[j]|, j in [hop*i, min(hop*(i+1), samples.Length))) * ampMult / count
```

i.e. window `[i*256, (i+1)*256)`, divided by the **actual** sample count (last frame clips), and
written for every frame including unvoiced ones. This is the writer convention KiraFrqGen should
follow (see below), and it is what `frq-format.md` already cites.

## What the frqeditor manual says about volume

- The table holds f0, key and volume; volume is one value per **256 samples** (4-1-5, p.138). Below
  ~172 Hz one period exceeds the window, so bar heights alternate large/small — expected, not
  corruption (4-1-5; `frq-format.md`).
- frqeditor **cannot edit volume**: "the volume part of the frequency table cannot be edited"
  (3-13-17); whether volume relates to table corruption is unknown and volume is explicitly out of
  scope for repair (4-2-1: "音量を編集できるツールはないため、検討範囲外とする").
- The graph shows the **table's** volume values; `pmk` is the exception — pmk has no volume, so
  frqeditor pulls it **directly from the wav** when converting `pmk -> frq` (3-9-1) and for the pmk
  editor display (3-13-17). No numeric scale is documented for either path.
- The auto-check uses volume to find "sound groups", but deliberately leaves the "voice present"
  threshold open — absolute or histogram-based, value to be tuned per source (5-7-6). So the check
  works on any consistent positive scale.
- Recommended workaround for quiet sources is to normalize a *copy* of the wav before generating
  the table, purely so the volume graph is easier to read (4-6-6..4-6-9). Distribution wavs should
  keep their natural dynamics. This implies KiraFrqGen must measure the actual distributed wav,
  not a normalized surrogate.
- The unedited "wild volume oscillation" pattern (large/small alternating or otherwise implausible)
  is not repairable in frqeditor (3-13-17, 4-1-4) — one more reason to write a clean envelope.

## Recommendation for KiraFrqGen

Compute, for every frame `i` in `0..N-1`, on the decoded WAV (mono; downmix stereo; resample to
44.1 kHz before framing if the input is not 44.1 kHz), with `x` as float samples in [-1, 1]:

```
lo = i * 256
hi = min(lo + 256, len(x))
amp[i] = 0                                    if hi <= lo
       = (2^15 / (hi - lo)) * Σ|x[j]|,  lo <= j < hi   otherwise
```

- This is exactly OpenUtau's `Frq.Build` (primary source above) and matches the SpeedWagon corpus
  writer's scale (median ratio 0.995, 99.6% of its files within 5%).
- Write an amplitude for **all** frames, voiced and unvoiced (the corpus writers and OpenUtau do;
  unvoiced frames are never forced to 0).
- Only the last frame needs the clipped window and the `hi - lo` divisor.
- Bounded `0..32768` (a full-scale sine gives ~20,861) — stable across files, monotone with level,
  no per-file normalization, so frqeditor's graph and its histogram-based group detection behave
  the same from bank to bank.
- On **check/repair of existing tables**, preserve the amplitude array verbatim (per the
  `frq-format.md` writing rule); only regenerate amplitude together with a full table generation.

If #8 later prefers maximal compatibility with the incumbent UTAU engines over OpenUtau
compatibility, the swap is the majority corpus convention:

```
amp[i] = (2^15 / sqrt(2)) * sqrt( Σ x[j]^2 / (hi - lo) )     // = rms(frame)/sqrt(2)*2^15
```

Both are within ~1.5x on voiced material, and frqeditor documents no scale, so this is a policy
choice, not a correctness one. The `mean(|x|)` form is recommended because it has a public
reference implementation, matches the dedicated frq generator (SpeedWagon), and is the convention
the repo's `frq-format.md` already names.

## Uncertainty / open questions

- Writer identities beyond SpeedWagon are inferred from behavior, not documented: the reserved
  bytes do not name the tool, and no resampler source for `fresamp14`/`resampler`/`phavoco` is
  available locally. The `44100`-writer's `3 * rms(1024)` is exact but its origin (and the 3x
  factor) is unexplained; fresamp14's variable analysis window (`f1` = 1024 samples) is a
  plausible but unverified candidate.
- SpeedWagon's exact envelope window was not reproduced to per-frame precision (scale is certain,
  window is not). Whether it matters for display is unknown because frqeditor's graph scale is
  undocumented.
- The orphan bank's writer (142 files) and 40 low-activity SpeedWagon files remain unclassified.
- All fit evidence comes from wavs that may have been edited after table generation (the manual
  actively recommends normalizing analysis copies), which limits per-frame exactness; per-file
  median scales are robust to this.

## Sources

- OpenUtau `OpenUtau.Core/Classic/Frq.cs` (master), `Frq.Build`:
  <https://github.com/openutau/OpenUtau/blob/master/OpenUtau.Core/Classic/Frq.cs>
- frqeditor manual `.local/manual.extract.txt`: 3-9-1 (l.1788), 3-13-17 (l.2332), 4-1-4 (l.3002),
  4-1-5 (l.3016), 4-2-1 (l.3173), 5-7-6 (l.5968), 4-6-6..4-6-9 (l.3780+); PDF in `.local/`.
- SpeedWagon (Custom.Maid, 2013-01-10, source distributed): <http://custom-made.seesaa.net/article/312531314.html>;
  listed as a frq-only generator in the UTAU音源制作wiki: <https://w.atwiki.jp/vbmaker/pages/54.html>
- 55 Hz unvoiced convention: oatsu's `utau_tools` frq wrapper comment (`f0 > 55` filter),
  <https://github.com/oatsu-gh/utau_tools>
- `docs/research/frq-format.md` (layout, reserved bytes, `mean(|x|)*2^15` note).

## Local verification

`.local/scan.py` + `.local/correlate.py` + `.local/classify.py` + `.local/final_stats.py` +
`.local/window_search.py` over the local corpus (4,654 frq, 4,512 paired):

- Reserved markers: zero 3,397; `" speedwagon     "` 895; `44100` 362.
- Formula strict counts (median within 2% of candidate): `meanabs*2^15` 683,
  `rms/sqrt(2)*2^15` 2,688, `rms*2^15` 420, `3*rms1024` 363, other 181.
- Example exact fits: one `55.0`-unvoiced-bank file — 915/915 frames within 1% of
  `rms256`; one low-pitched bank file — `a/rms1024` = 2.92–3.10 across all
  frames (median 2.995), `a/rms256` swings 2.6–5.7.
- `pmk_frq_check.py`: frq f0 matches pmk segments in only ~26–31% of frames in every group, so
  none of these frq files are pmk conversions.
