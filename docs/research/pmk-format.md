# `pmk` format (TIPS frequency table)

Status: **reverse-engineered and corpus-verified**. No public spec, parser, or source exists;
TIPS (by ScientistB) is closed source. Verified against all 1,231 `.pmk` files in the local corpus.

`.pmk` is TIPS's frequency table. Unlike `frq`, it stores **no volume**; f0 is quantized to a
period in samples. TIPS regenerates the file when its own version changes and force-regenerates
with the `R` flag. Redistribution of `.pmk` files is explicitly allowed in the TIPS readme.

## Binary layout (little-endian)

Header (14 bytes):

| Offset | Type | Field |
| --- | --- | --- |
| 0x00 | `int16` | version, `19` for TIPS 0.19β (0x13) |
| 0x02 | `float64` | average voiced code — **not Hz**; `44100 / avg` ≈ average key frequency in Hz |
| 0x0A | `int32` | entry count `N` |

Entries (`N` × 8 bytes):

| Type | Field | Meaning |
| --- | --- | --- |
| `int32` | `pos_end` | end position of the segment in samples (44.1 kHz timeline) |
| `int32` | `code` | period in samples; f0 ≈ `44100 / code` Hz |

- File size is exactly `14 + 8*N` in all 1,231 local files.
- **`code == 49` means unvoiced / no pitch.** Every local file uses 49 as its minimum, with huge
  counts of it (1.59M occurrences across the corpus). It is *not* a literal 900 Hz: TIPS 0.19β
  accepts voiced codes only in `50..=511` (f0 ≈ 86.3–882 Hz); a period that would round to 512 or
  higher is written as 49 instead. Black-box (#37): a hand-edited file containing a code `>= 512`
  makes TIPS produce no output wav at all, so the whole entry renders silent. Values 53–59 appear
  rarely. Other codes observed: 50–448 across the corpus (~98–882 Hz), the bulk of voiced entries
  in 120–300; the low end comes from low-pitched banks.
- `pos_end` is approximately cumulative: within a run of constant `code`, consecutive entries
  advance by exactly that code. At transitions the increments deviate slightly (±dozens of samples
  total per file) — the values appear pitch-mark aligned rather than recomputed. Do not
  `pos_end[i] = pos_end[i-1] + code[i]` when round-tripping; preserve stored values.
- `avg` field = mean of the voiced codes, verified across ~1,200 files (ratio ≈ 0.9999; outliers
  0.93). To display it as Hz: `44100 / avg`.
- The 2015 frqeditor manual's description ("entries are like squares whose step equals their
  value, pos_begin = previous pos_end") matches the observed structure closely enough for
  conversion, with the transition caveat above.

## Conversion semantics (as implemented by frqeditor, manual 3-9)

- `pmk -> other`: no amplitude data exists, so frqeditor reads amplitude directly from the wav.
- `other -> pmk`: unvoiced frames must become `code = 49`.
- All conversions between dio/frc/frq/vs4ufrq/pmk are lossy in detail (different frame grids and
  quantization); manual recommends converting once after editing, not round-tripping.

## Licensing

- TIPS: freeware, no license file, source not public. Redistributing `TIPS.exe` requires the
  author's permission; redistributing `.pmk` files is explicitly allowed.
- This spec is clean-room (derived from file formats alone), so no code obligations.

## Sources

- TIPS 0.19β readme (ScientistB): <http://scientistb.web.fc2.com/program/index.html> (no format docs)
- Black-box TIPS 0.19β runs on synthetic tables (#37): a code `>= 512` produces no output wav, 511
  renders; the local binary was only observed, never inspected.
- UTAU音源制作wiki 周波数表: <https://w.atwiki.jp/vbmaker/pages/54.html>
- An earlier clean-room draft was largely correct; corrections below.

## Corrections to the original reverse-engineering draft

1. `avg_f0` is in **code units**, not Hz (`44100 / avg` is the Hz value). It equals the mean of the
   voiced codes, not of the physical f0.
2. Unvoiced is represented as `code = 49` (not 0, not absent).
3. `pos_end` is only approximately `prev + curr_code`; transition entries deviate by a few samples.
4. Entry count can be verified against file size: `N == (size - 14) / 8`.

## Local verification

Scripts: `.local/pmk_scan2.py`, `.local/pmk_correlate.py`, `.local/pmk_debug.py`.

- All 1,231 files: `version == 19`, `size == 14 + 8*N`, min code 49.
- One C4 corpus file: avg=169.841 (→259.7 Hz ≈ C4), N=2350. Runs: code 49 from
  t=0.00–0.52 s, voiced codes 156–186 (267–237 Hz) until t=3.26 s, then 49 to the end (4.56 s).
  Within voiced runs the step equals the code; 209/2349 transitions deviate (mostly ±small).
- No zero or negative codes exist in the corpus; codes `< 60` occur only 67 times in 1.59M entries.
