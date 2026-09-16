# `frq` format (UTAU classic frequency table)

Status: **verified** against 4,650 real `.frq` files from a local UTAU corpus and multiple
independent open-source implementations.

The `.frq` file is the default UTAU frequency table, used by `resampler`, `fresamp`,
`phavoco`, `EFB-GT` (read-only), `WARP` (read-only) and produced by `SpeedWagon` and
`frqeditor`. It stores a per-frame f0 estimate plus a per-frame volume (amplitude) value.

## Binary layout (little-endian, no padding)

| Offset | Type | Field |
| --- | --- | --- |
| 0x00 | `char[8]` | magic `"FREQ0003"` (exact, no NUL) |
| 0x08 | `int32` | `hop` — samples between f0 frames. `256` in all 4,650 local files; UTAU assumes 44.1 kHz, so 5.8 ms |
| 0x0C | `float64` | average f0 / key frequency in **Hz** |
| 0x14 | `byte[16]` | reserved. Usually zero, but **do not assume**: 1,257 local files have non-zero reserved bytes - 362 store `44100` as int32 at 0x14 (`44 AC 00 00` = sample rate), 895 carry the space-padded ASCII marker ` speedwagon` (SpeedWagon provenance) |
| 0x24 | `int32` | frame count `N` |
| 0x28 | `float64[N]` | f0 per frame in Hz; `0.0` = unvoiced ("周波数なし") |
| 0x30 | `float64[N]` | amplitude per frame (raw scale, see below) |

- File size is exactly `40 + 16*N`. All 4,650 local files satisfied this; there is no trailer.
- The f0 and amplitude arrays are **interleaved** as pairs (`f0, amp, f0, amp, ...`), not two blocks.
  Parsers that read the whole tail as `f64` and slice `[::2]`/`[1::2]` work for the same reason.
- Unvoiced is `f0 == 0.0`; no negative/NaN/`0xFFFFFFFF` sentinel was found in any implementation or
  file. OpenUtau additionally treats `f0 <= 60 Hz` as unreliable in some code paths; `oatsu`
  tooling maps `<= 55` to unvoiced when converting from moresampler `mrq`.
- Amplitude is float64 but its scale is **writer-dependent**: local sample values range 0–7082;
  OpenUtau writes `mean(|x|) * 2^15` (0–32768); PyUtauCli writes 0–1. No resampler is known to
  consume it for synthesis; treat as display/relative data and preserve verbatim.
- The manual documents a separate quirk: at f0 <= ~172 Hz one 256-sample window is shorter than the
  pitch period, so consecutive amplitude bars alternate large/small. That is expected, not corruption.

## Reading robustness checklist

1. Check the magic; reject or convert anything that is not `FREQ0003` (no `FREQ0001/0002` is known).
2. Prefer `N` from the header, but cross-check against `(filesize - 40) / 16`; if they disagree,
   prefer the file size (truncated/patched files exist in the wild).
3. Accept both naming conventions for lookups: `<name>.wav.frq` and `<name>_wav.frq`
   (UTAU/OpenUtau generate `_wav.frq`; local corpus is 100% `_wav.frq`).
4. Do not assume `hop == 256`; read it (other rates could appear in theory). Time of frame `i` is
   `i * hop / 44100` under UTAU's fixed-rate assumption.
5. Writing: preserve the 16 reserved bytes and the amplitude array verbatim unless explicitly asked
   to regenerate them.

## Where the key frequency lives, and what it means

- The `float64` at 0x0C is the key/recording pitch used for UTAU's `mod` processing:
  additional pitch shift ≈ `(f0 - key) * mod / 100` (manual 2-4-1; engines may implement variants).
  Wrong key does not cause glitches, but it makes `mod != 0` out of tune.
- Not all engines read it: `EFB-GT`/`WARP` compute a median at synthesis; `VS4U`/`w4u` don't store
  it at all and compute a mean; `fresamp`/`model4`/`phavoco`/`resampler`/`TIPS` use the file value.
- frqeditor recomputes key on auto-repair: mean of voiced frames (manual 5-8-5).

## Unvoiced ("----") semantics per engine

From manual 2-5: `fresamp`/`phavoco`/`resampler` substitute the key frequency at unvoiced frames
(can still pitch-shift); `model4`/`TIPS`/`VS4U>=1.33`/`w4u` cannot pitch-shift unvoiced frames
(good for real unvoiced consonants, bad for vowels); `EFB-GT`/`WARP` may fail to render if
everything is unvoiced. Guideline: voiced regions should never be left without an f0; genuine
unvoiced consonant/breath regions should stay unvoiced.

## Sources

- OpenUtau: [`cpp/worldline/classic/frq.cpp`](https://github.com/openutau/OpenUtau/blob/master/cpp/worldline/classic/frq.cpp),
  [`OpenUtau.Core/Classic/Frq.cs`](https://github.com/openutau/OpenUtau/blob/master/OpenUtau.Core/Classic/Frq.cs)
- [titinko/frq_reader](https://github.com/titinko/frq_reader/blob/master/frq_reader.py)
- [PyUtauCli frq.py](https://github.com/delta-kimigatame/PyUtauCli/blob/master/PyUtauCli/voicebank/frq.py)
- [UtaUtaUtau/nnsvslabeling harvest_frq.py](https://github.com/UtaUtaUtau/nnsvslabeling/blob/main/harvest_frq.py)
- UTAU音源制作wiki 周波数表: <https://w.atwiki.jp/vbmaker/pages/54.html>
- masao's format diagram via UTAU DB: <http://utaudb.sakura.ne.jp/knowledge.php?id=6>

## Local verification

`.local/frq_scan2.py` + `.local/verify_formats.py` over `voice/**/*.frq`:

- 4,650/4,650 files: magic `FREQ0003`, hop 256, size `40+16*N`, count field matches file size.
- 1,257 files have non-zero reserved bytes: 362 store `44 AC 00 00` (44100) in the first 4 bytes,
  895 carry the space-padded ASCII marker ` speedwagon` (SpeedWagon-generated).
- Amplitude array scales per writer are analysed in [frq-amplitude-scale.md](frq-amplitude-scale.md).
- Example `aR_wav.frq`: N=690, avg=345.868 Hz, 256/690 frames unvoiced, amp up to 7082.8.
