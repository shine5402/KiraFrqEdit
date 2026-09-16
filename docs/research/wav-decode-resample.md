# WAV decode and resampling choices for KiraFrqGen

Status: research for map #1 / ticket #5. Crate facts were checked against crates.io and upstream
source checkouts on 2026-09-17; every version below is the latest published on that date. No
implementation exists yet.

KiraFrqGen must decode PCM WAV (16/24-bit int, 32-bit float, mono/stereo, 44.1 kHz and strays),
downmix to mono, and resample non-44.1 kHz input to 44.1 kHz before f0 estimation. The last step
is forced by the table formats, which carry no sample rate: `frq` stores a fixed 256-sample hop and
its consumers assume 44.1 kHz samples — OpenUtau hard-codes `wavSampleRate = 44100; // Sample rate
for .frq files is fixed` (`OpenUtau.Core/Classic/Frq.cs:132`) and throws on non-44.1 kHz WAVs in
the classic render path (`Worldline.cs:479-485`, this repo's `frq-format.md`). `pmk` quantizes f0
as a period in samples on the same 44.1 kHz timeline (`pmk-format.md`).

## Decoding

| Crate | Version (release) | License | Format coverage | Non-PCM (ADPCM, etc.) | API fit |
| --- | --- | --- | --- | --- | --- |
| [`hound`](https://crates.io/crates/hound) | 3.5.1 (2023-09-25) | Apache-2.0 | WAV only: PCM 8/16/24-bit int (packed 3-byte and 24-in-4), IEEE float32, `WAVE_FORMAT_EXTENSIBLE` PCM/float GUIDs; mono/multichannel | `WAVE_FORMAT_ADPCM` and unknown tags fail at open with `Error::Unsupported`; float bits != 32 is a `FormatError`; no 64-bit float | Smallest possible: `WavReader<R: Read + Seek>` over any reader (no mmap), on-demand `samples::<T>()` iteration, typed samples (`i16`, `i32`, `f32`), one `WavSpec` struct |
| [`symphonia`](https://crates.io/crates/symphonia) | 0.6.1 (2026-08-13) | MPL-2.0 | Full media framework. WAV demuxer: PCM u8..s32, 24-bit variants, f32/f64, A-law/µ-law, MS ADPCM and IMA ADPCM; plus FLAC/MP3/Ogg/MP4/MKV if ever needed | Unknown `wFormatTag` -> `unsupported_error("wav: unsupported wave format")`; supported-but-odd ADPCM variants are decoded (MS/IMA), malformed chunks -> `DecodeError` | Heavier: probe -> format reader -> decoder, packets into `AudioBuffer`/`GenericAudioBufferRef`, conversion via `copy_to_slice_interleaved`. Built for streaming and multi-codec apps |
| [`waveadapter`](https://crates.io/crates/waveadapter) | 0.2.0 (2026-08-05) | MIT OR Apache-2.0 | WAV only: U8/I16/I24_3/I24_4/I32/F32/F64, extensible headers; pure Rust, no C | Unsupported formats incl. ADPCM -> `WavError::UnsupportedFormat` (header can still be inspected) | Reads WAV into `audioadapter` buffers, i.e. the same buffer types rubato 5 uses; sibling crate by rubato's author (used in rubato's own `resample_wav` example); brand new and low-adoption (2 releases, ~2k downloads) |

Not considered credible alternatives: `wav` (header/metadata parsing, not a sample decoder) and
playback-oriented shims such as `rodio` (they wrap one of the above anyway).

Maintenance / error-reporting notes:

- `hound`: zero runtime dependencies, 16.8M downloads. Last release 2023, but upstream master is
  still receiving fixes (e.g. `9a11bf0`, 2026-08-09 "Saturate on overflow in write_sample"), so
  "stable, slow-moving" rather than abandoned. Errors are a small enum (`FormatError(&str)`,
  `Unsupported`, `InvalidSampleFormat`, `TooWide`, `IoError`), which is easy to map to per-file
  skip/report semantics.
- `symphonia`: actively developed (0.6.1 shipped 2026-08-13, MSRV 1.85), MPL-2.0 is file-level
  copyleft only (we never modify its files, so static linking is fine). Its error type
  (`symphonia_core::errors::Error`) carries enough detail for good user messages.

**Verdict: `hound` 3.5.1 for the primary path** — the requirement is exactly WAV/PCM, and hound
does that with no dependency footprint and the simplest streaming API. `symphonia` 0.6.1 is the
fallback decoder if ADPCM WAVs or other containers ever have to be accepted (it decodes ADPCM
rather than rejecting it). `waveadapter` is a watch item, not a dependency: too new, and its
`audioadapter` buffers only pay off if the resampler is also rubato 5.

## Resampling

| Crate | Version (release) | License | Method / quality | MSVC build friction | Thread safety | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| [`rubato`](https://crates.io/crates/rubato) | 5.0.0 (2026-08-10) | MIT OR Apache-2.0 | Pure Rust. `Async` = windowed-sinc interpolation with anti-aliasing (cutoff chosen automatically from filter length/window, `asynchro_sinc.rs:54-87`), plus a polynomial no-AA mode; `Fft` = synchronous FFT resampler with anti-alias filter, "always delivers the best quality" per README. Free ratios (160/147, 2/1, ...) | None: no C, no CMake; pure-Rust deps (`realfft`, `audioadapter`); MSRV 1.87 | `Resampler: Send` (`lib.rs:188`); intended per-stream/per-file instances, no globals; safe to construct inside each worker thread | **Recommended** |
| [`samplerate`](https://crates.io/crates/samplerate) | 0.2.4 (2021-01-14) | BSD-2-Clause (crate + bundled libsamplerate) | libsamplerate band-limited sinc (`SincBestQuality` / medium / fastest); classic SRC quality, but the vendored code is **libsamplerate 0.1.9 (2016-09-23)** | Always builds vendored C via `cmake` (`libsamplerate-sys` build.rs; crate resolves to 0.1.12, which still uses `cmake`): requires a CMake executable in addition to the MSVC C toolchain | `Samplerate` holds a raw `*mut SRC_STATE` -> not `Send`/`Sync`; one converter per thread works, but it cannot be shared/moved | Fallback only |
| [`soxr`](https://crates.io/crates/soxr) | 0.6.0 (2024-01-25) | LGPL-2.1 (wrapper); libsoxr itself LGPL | libsoxr VHQ/HQ, excellent quality | `libsoxr-sys` 0.1.2 links a **system** libsoxr through `pkg-config` — no vendoring; on Windows that means vcpkg/MSYS2 setup | Raw FFI, unsafe | Rejected: LGPL + Windows build friction |
| [`dasp_interpolate`](https://crates.io/crates/dasp_interpolate) | 0.11.0 (2020-05-29) | MIT OR Apache-2.0 | Per-frame interpolation primitives (`linear`, `sinc`), not a rate converter: no anti-alias lowpass when downsampling, no delay/edge management, no arbitrary-ratio engine | None (pure Rust, no deps) | No shared state | Rejected: wrong abstraction, dormant since 2020 |

### Quality and settings

- Both rubato paths are band-limited and explicitly anti-aliased. For an offline batch feeding f0
  estimation the `Fft` resampler is the right default: README's own guidance is "convert an audio
  file from one fixed rate to another ... as an offline batch job: use `Fft` with `process_all()`",
  and it costs no quality settings.
- For 44.1 <-> 48 kHz the ratio is 160/147; 96 -> 44.1 kHz is a 2/1-style downsample. `Fft::new`
  takes the two rates in Hz and derives the internal FFT sizes and the anti-alias cutoff from them
  (`synchro.rs:214-228`, cutoff formula `synchro.rs:97-103`), so no manual filter tuning is needed.
- Delay/edges: `Fft` has a `fft_size_out / 2` startup delay (`synchro.rs:692`), and `Async` an
  interpolation delay (`asynchro.rs:747`). `Resampler::process_all()` / `process_all_into_buffer()`
  run the whole clip and trim the delay, so the output lines up with the input (`rubato/README.md`,
  "Resampling a given audio clip"). This is exactly the offline use case.
- If a chunked path is ever wanted (progress reporting per chunk), `Async::new_sinc` with the
  default `SincInterpolationParameters::new(256, WindowFunction::BlackmanHarris2)` gives an
  automatic cutoff, `oversampling_factor = 128`, `Cubic` interpolation, and output quality the
  README describes as matching `Fft` (`asynchro_sinc.rs:26-87`).

### Version churn caveat

rubato shipped majors 3.0.0 (2026-05-20), 4.0.0 (2026-07-09) and 5.0.0 (2026-08-10); 5.0 moved to
the `audioadapter` buffer traits, so example code from before mid-2026 will not compile. Pin
`rubato = "5.0.0"` exactly and keep the wrapper (`audioadapter` adapters) behind our own
`decode_and_resample()` function so that a future major bump is a one-file change. This churn is
the main argument in favour of the otherwise stale `samplerate` crate; it does not outweigh rubato's
quality, purity and activity.

## What comparable tools do

| Tool | Decode | Resample | Evidence |
| --- | --- | --- | --- |
| OpenUtau (classic worldline) | NAudio `Core.Format.Wave` | **None.** `RegenFrq` loads samples at native rate and passes a hard-coded `44100` to `Worldline.F0(samples, 44100, stepMs, ...)` with `stepMs = 256 * 1000 / 44100` | `OpenUtau/ViewModels/SingersViewModel.cs:480-541` (master, fetched 2026-09-17); classic synth path throws `Unsupported sample rate ... Only 44100 Hz is supported` (`OpenUtau.Core/Render/Worldline.cs:479-485`); `Frq.cs:132` fixes `wavSampleRate` at 44100 |
| UtaUtaUtau `harvest_frq.py` | `soundfile` (libsndfile) | **None.** Runs `pyworld.harvest` at the file's own rate with `frame_period = 1000 * hop / fs`, i.e. a 256-sample hop on the native grid | repo `UtaUtaUtau/nnsvslabeling`, `harvest_frq.py` (master, fetched 2026-09-17) |
| PyUtauCli `voicebank/frq.py` | Python stdlib `wave` + manual int decoding (8/16/24/32-bit; **no float**; stereo takes the left channel) | **None.** `pw.harvest(datas, framerate, frame_period=1000 / framerate * 256)` | repo `delta-kimigatame/PyUtauCli`, `PyUtauCli/voicebank/frq.py:109-179` |
| putao (Python resampler) | `soundfile`, native rate | **None.** `pyworld.wav2world(*soundfile.read(...))`, `utils.srate` reads the actual rate | repo `ongyx/putao`, `putao/resamplers/world.py:38,57-67`, `putao/utils.py:64-67` |
| Rust UTAU tooling | — | — | crates.io search API for `utau` returns **zero crates**, `frq` returns nothing relevant (queried 2026-09-17) |

The pattern: the Python ecosystem sidesteps resampling by analyzing on the native grid. That is
only correct as long as the consumer is actually fed 44.1 kHz files; for stray rates it silently
produces a frq whose time base is wrong, and OpenUtau goes further and labels 48 kHz samples as
44.1 kHz to WORLD. KiraFrqGen resampling to 44.1 kHz first is therefore both the safer policy and
the one that matches the fixed-rate assumptions baked into the formats.

## Direct answer

> "resampling audio to switch between samplerate is kinda hard. do rust have some existing good
> cargo to do it, or should we just use libsamplerate?"

**Yes, Rust has a good crate for this: `rubato` — use it, don't take on libsamplerate.** It is pure
Rust (no C toolchain, no CMake on Windows/MSVC), MIT/Apache-2.0, maintained (5.0.0 in August 2026),
and its `Fft`/sinc resamplers are band-limited and anti-aliased; quality target is not a concern for
f0 estimation. The libsamplerate bindings (`samplerate`) are not wrong, just stale: the last release
is from 2021, it vendors **libsamplerate 0.1.9 (2016)** and always builds it through CMake
(`libsamplerate-sys`), and it hands back a raw FFI pointer. There is no quality or correctness
reason to prefer it in 2026; "just use libsamplerate" is the 2020 answer to this question.

## Recommendation (with versions)

- **Decoder: `hound = "3.5.1"`.** `WavReader::open` (any `Read + Seek`, no mmap) -> read
  `WavSpec`; `samples::<i32>()` for `SampleFormat::Int`, `samples::<f32>()` for `Float`; convert to
  `f32`/`f64` by dividing ints by `2^(bits-1)` (24-bit included; hound returns it in an `i32`);
  downmix stereo `(L + R) / 2` *before* resampling, and skip the resampler entirely when
  `sample_rate == 44100`. On `hound::Error::Unsupported`, report "ADPCM/non-PCM WAV; convert to
  PCM first" and move on.
- **Resampler: `rubato = "5.0.0"`** (default features) with the synchronous FFT resampler:

  ```rust
  let mut resampler = rubato::Fft::new(
      spec.sample_rate as usize, 44100, // Hz, fixed ratio
      2048, 1, rubato::FixedSync::Both, // chunk hint, mono
  )?;
  let needed = resampler.process_all_needed_output_len(samples.len());
  let out = resampler.process_all(&adapter_in, samples.len(), None)?;
  ```

  No quality settings to tune; `process_all` allocates the output and trims the resampler delay.
  For a chunked/streamed variant use `Async::new_sinc(44100.0 / rate, 1.0, &SincInterpolationParameters::new(256, WindowFunction::BlackmanHarris2), 1024, 1, FixedAsync::Input)`.
- **Threading:** one decoder + one resampler instance per file, constructed inside the worker
  thread. Nothing is global; `rubato`'s `Resampler` is `Send`, and hound readers are plain owned
  structs.
- **Fallback decoder:** `symphonia = "0.6.1"` with `default-features = false,
  features = ["wav", "pcm", "adpcm"]` if ADPCM or additional containers must be accepted.
- **Fallback resampler:** `samplerate = "0.2.4"` (`ConverterType::SincBestQuality`) only if rubato
  proves unusable; it then brings the CMake build dependency described above. `soxr` is rejected on
  license/build grounds.

## Sources

- crates.io API metadata and feature lists (queried 2026-09-17): `hound`, `symphonia`, `rubato`,
  `samplerate`, `libsamplerate-sys`, `soxr`, `libsoxr-sys`, `dasp_interpolate`, `waveadapter`;
  crates.io search `q=utau` (0 results), `q=frq` (no relevant crates).
- `hound`: <https://github.com/ruuda/hound> (master `9a11bf0`, 2026-08-09; WAV format-tag handling
  `src/read.rs:442-590`, ADPCM -> `Unsupported` `src/read.rs:453`, sample decoding
  `src/lib.rs:180-317`).
- `symphonia`: <https://github.com/pdeljanov/Symphonia> (master `ee35874`, 2026-08-12, v0.6.1;
  WAV tag dispatch `symphonia-format-riff/src/wave/chunks.rs:428-472`, PCM codec
  `symphonia-codec-pcm/src/lib.rs:186-260`, ADPCM `symphonia-codec-adpcm/src/{codec_ms,codec_ima_wav}.rs`;
  MSRV and license `README.md`).
- `rubato`: <https://github.com/HEnquist/rubato> (master `1d1da5c`, 2026-09-07, v5.0.0;
  `README.md` resampler-choice and clip-resampling sections; `src/synchro.rs:97-103,214-228,692`;
  `src/asynchro.rs:188-333,747`; `src/asynchro_sinc.rs:26-125`; `src/lib.rs:188,323,414-473`).
- `samplerate` / `libsamplerate-sys`: <https://github.com/Prior99/rust-samplerate> (master
  `70d1ad0`, 2021-01-14) and crates.io sources of `libsamplerate-sys` 0.1.12 (`build.rs`,
  `libsamplerate/NEWS` -> version 0.1.9, 2016-09-23; `libsamplerate/COPYING` BSD-2-Clause).
- `soxr` / `libsoxr-sys`: <https://github.com/haileys/soxr-rs> (master `c1a0dd2`, 2024-01-25),
  `libsoxr-sys` 0.1.2 `build.rs` (pkg-config probe).
- `dasp_interpolate`: <https://github.com/rustaudio/dasp> (last commit `d089297`, 2025-09-09;
  crate still 0.11.0, 2020-05-29).
- `waveadapter`: <https://github.com/HEnquist/waveadapter>, crates.io 0.2.0 (2026-08-05).
- OpenUtau: <https://github.com/OpenUtau/OpenUtau> (master, fetched 2026-09-17):
  `OpenUtau.Core/Classic/Frq.cs`, `OpenUtau.Core/Render/Worldline.cs`,
  `OpenUtau/ViewModels/SingersViewModel.cs`, `cpp/worldline/classic/resampler.cpp`.
- UtaUtaUtau harvest_frq.py: <https://github.com/UtaUtaUtau/nnsvslabeling> (`harvest_frq.py`).
- PyUtauCli: <https://github.com/delta-kimigatame/PyUtauCli> (`PyUtauCli/voicebank/frq.py`).
- putao: <https://github.com/ongyx/putao> (`putao/resamplers/world.py`, `putao/utils.py`).
