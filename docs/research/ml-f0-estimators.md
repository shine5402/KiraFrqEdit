# ML f0 estimators: candidates, licenses, runtimes and integration cost

Status: research for wayfinder ticket [#36] (2026-09-19), branch
`research/ml-f0-estimators`. Answers "which permissively-licensed ML f0 estimator(s) can meet
the destination, and what does adopting one cost?". Extends
[f0-generation.md](f0-generation.md) (the WORLD plan) and builds on the estimator seam and
timings in [world-integration.md](world-integration.md).

Conventions: **verified** = checked this session against the source that owns the claim (repo
files, release assets, HuggingFace API, ONNX model bytes); **measured** = run on this machine
(Windows, ONNX Runtime 1.30.0 CPU, Python 3.14; or the vendored WORLD v1.0.1 via
`kirafrq-world-binding`); **claimed** = the model author's or paper's own number, not reproduced
here. Sizes are exact bytes from the owning API. The destination is `F0Estimator` in
`crates/kirafrqgen-core/src/lib.rs`, the hop-256 (5.805 ms) `FrequencyTable`, and the 71-800 Hz
policy.

[#36]: https://github.com/shine5402/KiraFrqEdit/issues/36

## Summary

- **RMVPE is the best destination match** and is adoptable: Apache-2.0 code and an MIT-tagged
  weight mirror, a mature ONNX export, and **measured ~37-41x realtime on CPU** (vs Harvest's
  ~5x and DIO's ~37-56x in this repo). It is the de-facto f0 backend of the RVC ecosystem and is
  vocal/expressive-tuned, which is exactly where UTAU tables are weakest. Its cost is the
  **362 MB ONNX (or 181 MB `.pt`)** file and a 16 kHz log-mel input contract.
- **SwiftF0 is the cheap alternative**: MIT, **389 KB** model, raw-audio input, **measured
  ~47-56x realtime**, and the best overall score on the public pitch-benchmark. Weaker than RMVPE
  on expressive singing (Vocadito 92.6% vs 96.4%). Its ONNX bakes STFT into the graph, so it
  needs ONNX Runtime (`ort`), not `tract`.
- **CREPE (MIT)** and **FCPE (MIT)** are credible fallbacks/ensemble members. CREPE is
  tract-compatible (verified); FCPE is ~42 MB and octave-stable. **PESTO is LGPL-3.0 - exclude.**
- Runtime: use **`ort` (ONNX Runtime)**, which already has a worked reference implementation for
  all four models (`pitch-core-onnx`). Pure-Rust `tract` runs CREPE but **fails on SwiftF0's
  embedded STFT** (measured); RMVPE's GRU is unverified on tract.
- **Honest gap:** no primary source compares ML estimators head-to-head against WORLD Harvest on
  UTAU voicebank material. The case for ML is indirect (model noise benchmarks + Harvest's
  documented fragility), not a direct measurement; see "Voiced/unvoiced fidelity".

## Candidates and licenses

| Model | Code license | Weights license / mirror | Model file (exact bytes) | Native contract |
| --- | --- | --- | --- | --- |
| RMVPE | Apache-2.0 ([Dream-High/RMVPE][rmvpe-repo]) | MIT-tagged mirror ([lj1995/VoiceConversionWebUI][rmvpe-hf]); repo ships **no** weights | `rmvpe.onnx` 361,688,443 B; `rmvpe.pt` 181,184,272 B | 16 kHz log-mel, 128 mels, hop 160 |
| CREPE | MIT ([marl/crepe][crepe-repo]) | MIT: torchcrepe (MIT) and onnxcrepe (MIT) conversions | ONNX full/large/medium/small/tiny = 88,984,790 / 51,012,147 / 23,525,293 / 6,524,192 / 1,955,762 B ([onnxcrepe v1.1.0][onnxcrepe-rel]) | 16 kHz, 1024-sample frames, hop 160 |
| torchfcpe / FCPE | MIT ([CNChTu/FCPE][fcpe-repo]) | MIT (checkpoint committed in the repo) | `fcpe_c_v001.pt` 43,363,260 B; prebuilt ONNX ~42 MB ([pitch-core][pitch-core-models]) | 16 kHz raw audio, hop 160 |
| SwiftF0 | MIT ([lars76/swift-f0][swiftf0-repo]) | MIT (`swift_f0/model.onnx` committed) | `model.onnx` 397,987 B (PyPI 0.1.2 wheel: 399,114 B) | 16 kHz raw audio, hop 256 |
| PESTO | LGPL-3.0 ([SonyCSLParis/pesto][pesto]) | LGPL-3.0, no separate weights grant | ~17 MB ONNX | 16 kHz HCQT |

All byte counts above are **verified** through the GitHub contents/releases API, the HuggingFace
models API (`?blobs=true`), or `Get-Item` on the downloaded file.

### RMVPE

- Paper: Wei et al., *RMVPE: A Robust Model for Vocal Pitch Estimation in Polyphonic Music*,
  Interspeech 2023 ([arXiv:2306.15412][rmvpe-paper]). Architecture: deep U-Net + GRU over
  log-mel, predicting a `T x 360` pitch-probability matrix via local-argmax decoding. Trained to
  extract vocal f0 **directly from polyphonic music** (no source separation), and the paper claims
  robustness across SNR levels.
- **The official code repo publishes no weights** (verified: the `main` tree is `.py` + LICENSE
  only; no releases/tags). The weights adopted by the ecosystem are `rmvpe.pt`/`rmvpe.onnx` from
  the RVC project's HuggingFace repo `lj1995/VoiceConversionWebUI`, whose card is tagged
  `license:mit`. The RVC WebUI repo itself is MIT (verified). So the practical,
  **redistributable** weight path is the MIT mirror; note the provenance caveat that the original
  authors did not attach a weights license themselves. `pitch-core` states the same split:
  "Apache-2.0 code / MIT mirror" ([pitch-core MODELS.md][pitch-core-models]).
- Runtime shape (verified from the Rust reference implementation in `pitch-core-onnx::rmvpe`,
  which mirrors `rvc/lib/rmvpe.py`): 16 kHz mono; log-mel with `n_fft=1024`, `hop=160`,
  `n_mels=128`, `fmin=30`, `fmax=8000`; frames padded to a multiple of 32; output `[T, 360]` at
  10 ms per frame. Frequency is the weighted mean over +/-4 bins around the argmax; confidence is
  the peak salience; `rmvpe-onnx` zeroes pitch below a **0.03** confidence threshold.

### CREPE

- Kim et al., ICASSP 2018. CNN over 1024-sample time-domain frames; 360 pitch bins at 20 cents,
  nominal **~32.7-2006 Hz** ([pitch-core-onnx::crepe][pitch-core-rs]).
- Original weights ship inside the `crepe` PyPI package; the repo is MIT (verified). Conversions:
  `torchcrepe` (MIT, `.pth`: tiny 1,962,363 B, full 88,991,291 B) and `onnxcrepe` (MIT, ONNX as
  in the table). All capacities share one I/O contract: input `frames [n, 1024]` float32,
  per-frame zero-mean/unit-std normalized; output `probabilities [n, 360]`; default hop 160
  (10 ms). `torchcrepe` defaults to Viterbi decoding (fewer half/double-pitch errors) and
  recommends a periodicity threshold around **0.21** for clean speech.

### torchfcpe / FCPE

- Luo et al., *FCPE: A Fast Context-based Pitch Estimation Model* ([arXiv:2509.15140][fcpe-paper]).
  Lynx-Net (depthwise separable convolutions) over log-mel, `[T,128] -> [T,360]`, local-argmax
  decode. Paper claims **96.79% RPA on MIR-1K** and RTF **0.0062** on one RTX 4090 (GPU). The
  Rust reference says "5.3x faster than RMVPE, 77x faster than CREPE" ([pitch-core-onnx::fcpe][pitch-core-rs]).
- Code and the bundled checkpoint are MIT (verified: repo LICENSE MIT, `torchfcpe/assets/fcpe_c_v001.pt`
  committed). The ONNX export takes raw audio `[1, n, 1]` at 16 kHz and emits f0 `[1, n, 1]` with
  **0 = unvoiced** (the model gates internally at threshold 0.006), so the export exposes no
  voicing logits - confidence is binary in that wrapper.

### SwiftF0

- Nieradzik, 2025 ([arXiv:2508.18440][swiftf0-paper]). STFT + 2D CNN, **95,842 parameters**;
  paper claims **~42x faster than CREPE on CPU** and 91.80% harmonic mean at 10 dB SNR. The
  project's own benchmark repo claims **90x faster than CREPE** and rank #1 overall (90.2%) on
  8 datasets ([lars76/pitch-benchmark][pitch-benchmark]) - the two "faster" factors differ by
  machine, both author-owned (**claimed**).
- Verified contract (`swift_f0/core.py`): input raw audio `[1, N]` at 16 kHz, STFT `n_fft=1024`,
  `hop=256` (16 ms), 384-sample symmetric padding; outputs `pitch_hz` and `confidence`; nominal
  **46.875-2093.75 Hz**; default voiced threshold **0.9**. Model file is committed; repo MIT.

### PENN (noted, not adopted)

The public benchmark also scores PENN (best on noisy PTDB, 76.4%), but we found no permissive
weight grant in primary sources, so it is not treated as a candidate here.

## Runtime options from Rust

| Option | License | Fit | Notes |
| --- | --- | --- | --- |
| [`ort`][ort] (ONNX Runtime) | MIT OR Apache-2.0 | **Preferred** | v2.0.0-rc.13 wraps ONNX Runtime 1.28; default features `download-binaries` + `copy-dylibs` fetch/copy the right ORT per platform; `load-dynamic` allows runtime dylib loading. Windows/macOS/Linux CPU + execution providers. |
| [`tract`][tract] | MIT OR Apache-2.0 | Partial | Pure Rust, no C++/FFI, has a streaming "pulsification" mode. 0.23.4 **measured**: runs `crepe_tiny.onnx`, but **fails to analyze SwiftF0's ONNX** (`Failed analyse for node "/STFT" STFT ... inputs[0].rank == 3`, model input is rank 2). RMVPE's GRU op not exercised. |
| [`candle`][candle] | Apache-2.0 | Not drop-in | A tensor framework, not an ONNX loader; using it means reimplementing each architecture and converting weights. |
| `tch` / libtorch | MIT/Apache (bindings), BSD-3 (libtorch) | Not preferred | Heavy (~GB) native dependency; `ort` already covers exported models. |

- **Reference integration exists.** The third-party [`pitch-core-onnx`][pitch-core-crate] crate
  (MIT/Apache-2.0, by gzivdo) implements all four permissive backends on `ort` with exactly the
  streaming contracts above; it is strong evidence the integration is tractable and gives a
  template for mel/STFT preprocessing in Rust.
- ONNX Runtime itself is MIT ([onnxruntime.ai][ort-ep]); using it does not add a copyleft
  obligation. If ONNX Runtime is linked statically, its MIT notice must be retained; if shipped
  as a dylib via `ort`, it is a separate component.
- **`tract` caveat:** because some exports bake STFT into the graph (SwiftF0) or use GRU (RMVPE),
  a tract-only path would require re-exporting with feature extraction kept on the Rust side
  (feed frames/mel directly), plus per-model op verification. `ort` avoids this.

## CPU cost vs Harvest and DIO

**Measured** on this machine (Windows; ONNX Runtime 1.30.0 CPU, Python wrappers; WORLD via
`kirafrq-world-binding`, 71-800 Hz, 5 ms frame period for WORLD / native hops for ML). Realtime
factor = audio seconds / wall seconds. The 4.56 s row is one real voicebank sample; the 5 s and
30 s rows are a 10-harmonic 220 Hz synthetic (synthetic is harmonic-rich because Harvest needs a
spectrally rich signal - see the gotcha in [world-integration.md](world-integration.md)).

| Input | DIO | Harvest | SwiftF0 | RMVPE |
| --- | --- | --- | --- | --- |
| real 4.56 s corpus wav | - | - | 85-98 ms (**~47-54x**) | 104-152 ms steady (**~37x**); first call 625 ms (session init) |
| synthetic 5 s | 90 ms (55.7x) | 1020 ms (4.9x) | 92 ms (54.4x) | 220 ms (22.7x) |
| synthetic 30 s | 814 ms (36.9x) | 5573 ms (5.4x) | 532 ms (56.4x) | 724 ms (41.4x) |

- **Both ML candidates are comfortably faster than Harvest** (~5x) and in DIO's band
  (~37-56x); SwiftF0 edges DIO on the 30 s run. The repo's WORLD numbers (Harvest ~5x, DIO
  ~49-66x) are the reference ([world-integration.md](world-integration.md)).
- RMVPE's numbers vary run-to-run (22.7x at 5 s vs 41.4x at 30 s) because of ORT thread warmup
  and scheduling; treat "tens of times realtime" as the honest figure. SwiftF0's Python wrapper
  pins `intra_op_num_threads = 1` / `inter_op_num_threads = 1`.
- **Session init is not free:** the first RMVPE inference (graph optimization + weight load)
  took ~625 ms; load once and reuse the session across files.

## Input contract and output shape

| Model | Audio in | Preprocessing | Native output | Voicing signal |
| --- | --- | --- | --- | --- |
| RMVPE | 16 kHz mono | log-mel 128, `n_fft=1024`, hop 160, 30-8000 Hz, pad to x32 | f0 + salience `[T,360]`, 10 ms grid | confidence = peak salience; threshold ~0.03 |
| CREPE | 16 kHz mono | frames `[n,1024]`, per-frame mean/std normalize | probabilities `[n,360]`, 10 ms grid | sigmoid peak; threshold ~0.21 |
| FCPE | 16 kHz mono | raw audio `[1,n,1]` | f0 `[1,n,1]`, 10 ms grid, 0 = unvoiced | internal gate (0.006); binary after export |
| SwiftF0 | 16 kHz mono | raw audio `[1,N]`; STFT inside graph | `pitch_hz`, `confidence`; 16 ms grid | confidence threshold 0.9 |

All four want **16 kHz mono**; `kirafrq-audio` normalizes to **44.1 kHz mono**, so an ML
estimator must resample (a 44.1 -> 16 kHz step; `rubato` is already a dependency of
`kirafrq-audio`). None of the native grids is the UTAU 256-sample grid.

## Mapping onto `F0Track` / `FrequencyTable` (hop 256, 5.805 ms, 71-800 Hz)

The table builder (`crates/kirafrqgen-core/src/table.rs`) zips `track.f0_hz` onto table frame
indices 1:1 and treats non-finite or `<= 0` as unvoiced; the estimator is expected to deliver a
track already sampled at `frame_period_ms = 256/44100*1000 = 5.805 ms` with
`temporal_positions[n] = n * 5.805 ms` (this is what `WorldEstimator` does by passing the period
into WORLD). An ML estimator therefore needs:

1. **Resample the native contour onto the 5.805 ms grid.** Native hops are 10 ms (RMVPE, CREPE,
   FCPE) or 16 ms (SwiftF0); 256 samples at 44.1 kHz is 92.88 samples at 16 kHz, so no native
   hop aligns exactly - interpolate. Since 5.805 < 10 ms this is upsampling; linear (or
   log-f0) interpolation on `temporal_positions` is the analogue of WORLD's internal 1 ms grid
   being resampled to `frame_period`. Do not change the table-side `zip` semantics.
2. **Apply the 71-800 Hz policy as a post-filter.** Every candidate covers the range with margin
   (RMVPE/CREPE/FCPE ~32.7-1975/2006 Hz, SwiftF0 46.875-2093.75 Hz), so map out-of-range f0 to
   `0.0` (unvoiced), matching WORLD's floor/ceiling semantics. Note that UTAU tables are allowed
   to exceed 800 Hz (a corpus sample reached 858.57 Hz per [world-integration.md](world-integration.md));
   800 is policy, not a format limit, so keep it configurable (`F0Config::ceiling_hz`).
3. **Threshold confidence to voiced/unvoiced.** Only RMVPE, CREPE and SwiftF0 expose a
   usable confidence; FCPE's export is already gated. Whatever the threshold, it belongs in
   `F0Config` next to `floor_hz`/`ceiling_hz`, not hard-coded.
4. **`refine_stonemask` is a no-op** for ML estimators (the trait's default impl). The pipeline
   calls it whenever `opts.f0.stone_mask` is true (`run.rs`), so `stone_mask` must be forced off
   (or ignored) for an ML variant. `F0Track` needs no new fields for the f0 contour.

## Threading, parallelism and footprint

- The run pipeline already parallelizes whole files on a rayon pool sized by `jobs`
  (`build_pool` in `crates/kirafrqgen-core/src/run.rs`). ONNX Runtime sessions are thread-safe
  for concurrent `Run`, but ORT's **default intra-op thread pool plus the rayon pool will
  oversubscribe the CPU**. Set `intra_op_num_threads = 1` and `inter_op_num_threads = 1` on the
  session and let `jobs` be the parallelism knob (SwiftF0's own wrapper does exactly this; the
  `pitch-core-onnx` sessions do not override it, a latent inefficiency).
- **Memory/disk:** RMVPE is the outlier at 362 MB ONNX / ~180 MB `.pt`; CREPE-full ~89 MB;
  FCPE ~42 MB; CREPE-tiny ~2 MB; SwiftF0 ~0.4 MB. A single shared session per process is
  preferable to per-worker sessions.
- **Weights are not in the repo and should not be bundled by default** if repo size matters; a
  first-run download (with a license/attribution notice) is the usual pattern, and the MIT
  mirrors above make redistribution legal if we choose to bundle.

## Voiced/unvoiced fidelity on noisy / breathy regions

There is **no primary source that pits an ML estimator against WORLD Harvest on UTAU voicebank
audio**, so this section separates what is actually evidenced from what is inferred.

- **Model-side noise robustness (evidenced).** The public [pitch-benchmark][pitch-benchmark]
  reports harmonic-mean accuracy per dataset, including PTDBNoisy (speech + real background
  noise): **SwiftF0 74.0%, RMVPE 68.5%, CREPE 53.8%**, against ~90% clean. On singing sets RMVPE
  leads (MIR-1K 96.0%, Vocadito 96.4%); overall SwiftF0 leads (90.2%). These are **claimed**
  numbers owned by that benchmark.
- **Octave stability (evidenced).** `pitch-core`'s own benchmark (MIR-1K, Vocadito, 12 real
  songs) puts octave-error rate at **RMVPE 0.0-0.1%, FCPE 0.0-0.8%, CREPE 0.7-2.3%, SwiftF0
  0.8-3.3%**, and groups RMVPE/FCPE as "vocal-tuned" (capture vibrato/glissando/breath
  transitions more aggressively; cross-class agreement with CREPE falls from 98.5% on clean
  karaoke to 18.9% on separator-extracted vocals) ([pitch-core ANALYSIS.md][pitch-core-analysis]).
- **Voicing is entangled for RMVPE (evidenced).** `pitch-core` notes RMVPE's voicing comes from
  pitch confidence and "fails on short/quiet fragments", recommends taking voicing from a
  better-calibrated backend, and ships per-backend confidence recalibration (+3-17 pp voicing
  F1). This is a caution for the U/V decision, which is exactly what UTAU tables encode.
- **WORLD-side fragility (measured here).** Re-running the vendored WORLD v1.0.1 on synthetic
  signals reproduces the documented pure-tone gotcha: **Harvest 0/517 voiced on a 220 Hz sine
  (DIO 516/517)**. On harmonic signals with white noise up to 30%, both DIO and Harvest stay
  fully voiced, so white-noise-added harmonics do **not** expose a Harvest V/UV weakness.
- **What this means.** The defensible claim is: ML models are trained with noise augmentation
  and score well on noisy speech/singing benchmarks, while Harvest is documented to collapse on
  spectrally poor input. That is *suggestive* that ML gives more faithful U/V on breathy,
  aperiodic or partly-polyphonic regions, but it is **not** a direct measurement against Harvest
  on UTAU material. Treat "does ML beat Harvest on real breathy frames?" as an open validation
  item (below), not an established fact.

## What adopting one costs

Recommended first target: **RMVPE via `ort`**, with SwiftF0 as the small-footprint alternative.

1. New crate/module behind the existing seam, e.g. `kirafrq-ml-binding`, exposing an
   `F0Estimator` impl; all `ort`/`unsafe`-adjacent code confined there, matching the
   `kirafrq-world-binding` layout.
2. Add `ort` (MIT/Apache-2.0) with `download-binaries`; pin the ORT version deliberately, as the
   repo already does for `hound`/`rubato`/`cc`.
3. Reproduce RMVPE preprocessing in Rust (16 kHz resample, log-mel 128 x `n_fft=1024` x
   hop=160, 30-8000 Hz, x32 pad) or SwiftF0's raw-audio path; `pitch-core-onnx` is a direct
   reference (and MIT/Apache-2.0, so copyable with attribution if desired).
4. Contour resample to the 5.805 ms grid, then floor/ceiling and confidence threshold as
   `F0Config` fields; force `stone_mask` off for ML estimators.
5. Session created once, `intra/inter_op = 1`; weight acquisition + attribution notice per the
   license table; extend the CLI/GUI estimator selector (`Estimator` enum) with the ML choice.
6. Validation artifact: run RMVPE/SwiftF0 and Harvest/DIO over a sample of real voicebank wavs,
   compare voiced/unvoiced transitions and f0 error (e.g. against existing `_wav.frq`), with
   particular attention to breath and note transitions - this closes the open question above.

## Open questions

- Direct ML-vs-Harvest U/V comparison on real UTAU breathy material (no primary source exists).
- Whether RMVPE's confidence, recalibrated, is a better U/V signal than a hybrid
  (RMVPE pitch + a calibrated voicing source), as `pitch-core` suggests.
- tract compatibility for RMVPE's GRU and for an STFT-free SwiftF0/CREPE export, if a pure-Rust
  runtime is ever required.
- Weight mirror stability: RMVPE depends on a community HF mirror under a blanket MIT tag; a
  vendored copy with recorded hashes would de-risk it.

## Sources

- [Dream-High/RMVPE][rmvpe-repo] (Apache-2.0 code, no weights) and [arXiv:2306.15412][rmvpe-paper].
- [lj1995/VoiceConversionWebUI][rmvpe-hf] (`rmvpe.pt` 181,184,272 B, `rmvpe.onnx` 361,688,443 B;
  `license:mit` tag) and [RVC-Project/Retrieval-based-Voice-Conversion-WebUI][rvc] (MIT;
  `rvc/lib/rmvpe.py` preprocessing/threshold).
- [marl/crepe][crepe-repo] (MIT), [maxrmorrison/torchcrepe][torchcrepe] (MIT; `.pth` sizes),
  [yqzhishen/onnxcrepe][onnxcrepe] v1.1.0 (MIT; ONNX sizes and shared I/O contract).
- [CNChTu/FCPE][fcpe-repo] (MIT; checkpoint) and [arXiv:2509.15140][fcpe-paper].
- [lars76/swift-f0][swiftf0-repo] (MIT; `swift_f0/model.onnx`) and [arXiv:2508.18440][swiftf0-paper].
- [SonyCSLParis/pesto][pesto] (LGPL-3.0 - excluded).
- [lars76/pitch-benchmark][pitch-benchmark] (MIT; per-dataset accuracy incl. PTDBNoisy).
- [gzivdo/pitch-core][pitch-core-crate] (MIT/Apache-2.0; `pitch-core-onnx` Rust reference,
  MODELS.md license table, ANALYSIS.md accuracy/voicing notes).
- [pykeio/ort][ort] (MIT OR Apache-2.0; linking guide), [sonos/tract][tract],
  [huggingface/candle][candle], [ONNX Runtime execution providers][ort-ep].
- Local measurements: `kirafrq-world-binding` (vendored WORLD v1.0.1) and ONNX Runtime 1.30.0
  CPU; timings and V/UV counts recorded in this session. See
  [world-integration.md](world-integration.md) for the WORLD reference numbers.

[rmvpe-repo]: https://github.com/Dream-High/RMVPE
[rmvpe-paper]: https://arxiv.org/abs/2306.15412
[rmvpe-hf]: https://huggingface.co/lj1995/VoiceConversionWebUI
[rvc]: https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
[crepe-repo]: https://github.com/marl/crepe
[torchcrepe]: https://github.com/maxrmorrison/torchcrepe
[onnxcrepe]: https://github.com/yqzhishen/onnxcrepe
[onnxcrepe-rel]: https://github.com/yqzhishen/onnxcrepe/releases/tag/v1.1.0
[fcpe-repo]: https://github.com/CNChTu/FCPE
[fcpe-paper]: https://arxiv.org/abs/2509.15140
[swiftf0-repo]: https://github.com/lars76/swift-f0
[swiftf0-paper]: https://arxiv.org/abs/2508.18440
[pesto]: https://github.com/SonyCSLParis/pesto
[pitch-benchmark]: https://github.com/lars76/pitch-benchmark
[pitch-core-crate]: https://github.com/gzivdo/pitch-core
[pitch-core-models]: https://github.com/gzivdo/pitch-core/blob/main/MODELS.md
[pitch-core-analysis]: https://github.com/gzivdo/pitch-core/blob/main/ANALYSIS.md
[pitch-core-rs]: https://docs.rs/pitch-core-onnx/latest/pitch_core_onnx/
[ort]: https://github.com/pykeio/ort
[ort-ep]: https://onnxruntime.ai/docs/execution-providers/
[tract]: https://github.com/sonos/tract
[candle]: https://github.com/huggingface/candle
