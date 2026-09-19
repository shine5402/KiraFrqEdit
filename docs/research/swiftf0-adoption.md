# SwiftF0 adoption: `ort` runtime, weight provenance and integration delta

Status: research for wayfinder ticket [#59] (2026-09-19), branch `research/swiftf0-adoption`.
Answers the adoption questions beside the RMVPE path from [#36] (survey) and [#57] (provider
ticket): does `ort` execute the committed ONNX despite the in-graph STFT that `tract` failed on,
is the weights grant clean, what does integration cost over the planned RMVPE path, and what does
public evidence actually cover for UTAU-style material. It does **not** run a corpus comparison -
that stays a prototype, gated on the verdict below.

Conventions (mirroring [ml-f0-estimators.md](ml-f0-estimators.md)): **verified** = checked this
session against the source that owns the claim (repo files, commit history, ONNX bytes, ONNX
Runtime kernel tables, PyPI metadata); **measured** = run this session on this machine (Python
ONNX Runtime 1.28.0 and 1.30.0 CPU on synthetic signals); **claimed** = the author's or paper's
own number, not reproduced. Sizes are exact bytes; hashes are SHA-256.

[#36]: https://github.com/shine5402/KiraFrqEdit/issues/36
[#47]: https://github.com/shine5402/KiraFrqEdit/issues/47
[#48]: https://github.com/shine5402/KiraFrqEdit/issues/48
[#49]: https://github.com/shine5402/KiraFrqEdit/issues/49
[#53]: https://github.com/shine5402/KiraFrqEdit/issues/53
[#56]: https://github.com/shine5402/KiraFrqEdit/issues/56
[#57]: https://github.com/shine5402/KiraFrqEdit/issues/57

## Summary

- **Runtime: yes, `ort` runs it.** `ort` v2.0.0-rc.13 (the version [#48] fixed for the provider)
  bundles ONNX Runtime **1.28.0** (verified from `ort-sys`'s `dist.tsv` at the rc.13 tag), and
  ORT 1.28.0's CPU execution provider implements the `STFT` op (opset 17+). The committed model
  (opset 20) executes with identical outputs on ORT 1.28.0 and 1.30.0 - **bit-exact** across the
  five synthetic signals tested. `tract`'s STFT limitation is irrelevant on this runtime.
  Confidence: high (direct execution, not op-table inference).
- **Provenance: clean.** The model is committed by the author in the same MIT-licensed repo as
  the code (`swift_f0/model.onnx`, initial commit 2025-07-08, revised 2025-09-02); PyPI declares
  `license_expression: MIT` and bundles the same model. No third-party mirror, no
  weights-license gap - the caveat [#36] recorded for RMVPE does not apply here. MIT also makes
  bundling the 398 KB model legally clean if a later decision wants that.
- **Integration delta vs the [#57] RMVPE path is small and mechanical**: no mel preprocessing, no
  Rust-side decoder (the graph already emits `pitch_hz` and `confidence`), same `ort` runtime,
  same session/threading/resolution machinery, same 16 kHz resample, same #47/#53 output policy.
  New work is a raw-audio adapter, a **mandatory zero-pad to >=256 samples** (shorter inputs make
  ORT fail inside STFT), a per-model native-grid descriptor (**hop 256, +127.5-sample frame
  origin**), the 0.9 confidence default, and enum/UI wiring.
- **Evidence**: SwiftF0 leads the author-owned benchmark overall (90.2% vs RMVPE 87.2%) and on
  noisy speech (PTDBNoisy 74.0% vs 68.5%), but RMVPE leads solo expressive singing (Vocadito
  92.6% vs 96.4%); there is **no independent SwiftF0-vs-RMVPE comparison** and **no UTAU
  material** in any public source. The paper itself states its confidence scores "primarily
  reflect certainty in pitch class, not voicing decisions".
- **Verdict: a bounded corpus prototype is warranted**, but only after the provider lands and
  only to answer the two questions public evidence cannot: voiced/unvoiced behavior on real
  breath/plosive/noise-floor frames, and octave/half-double behavior on UTAU vibrato. The model
  is 398 KB, MIT, needs no new dependency, and reuses the entire ML seam, so the prototype is
  cheap; shipping it blind is not justified by public evidence alone.

## Model artifact and provenance

**Verified** from the upstream repository at its current `main` commit
`64700fce8ef39c2970814bf427ac1d75a2f20d72` (2025-09-02, "Chore: remove redundant ONNX nodes to
simplify model graph (Fixes #2)") - that commit *is* `main`, so pinning the commit pins the
cleaned graph:

| Artifact | Bytes | SHA-256 | Notes |
| --- | --- | --- | --- |
| `swift_f0/model.onnx` @ `64700fce` | 397,987 | `7e2390db8379cd9e1e2b22828e55b45b57c8559e4c8335678c717dc245c18176` | downloaded raw this session |
| `model.onnx` in PyPI `swift-f0` 0.1.2 wheel | 399,114 | `fa91bb45512b90339cf4b00a599ba8fe3a253c46419fcfe6b46df77a8a8336a5` | older export, extra dynamic-pad nodes |

- Both files: ONNX `ir_version` 9, opset `ai.onnx` **20**, producer `pytorch 2.7.0`, no
  `metadata_props`, no custom domains, input `input_audio` `[1, audio_length]` float32, outputs
  `pitch_hz` and `confidence` `[1, time_frames]` float32.
- The two exports are **bit-exact on every synthetic signal tested** (see Runtime below) despite
  different graph formulations: the repo model pads with a static `[0, 384, 0, 384]` initializer,
  the wheel model computes the same padding dynamically (`ConstantOfShape`/`Cast`, 34 extra
  `Constant` nodes). Output frame counts match for all lengths swept. The version difference is
  file layout, not math; pin the repo commit anyway.
- Op inventory (repo model): 26 standard ops - `STFT` x1, `Conv` x6, `Pad`, `Transpose`,
  `Reshape`, `Softmax`, `ArgMax`, `Gather`, `ReduceSum`, `Slice`, `Sqrt`, `Log`, `Abs`,
  `LessOrEqual`, plus scalar arithmetic. `STFT` attributes: `frame_step = 256` (int64),
  `frame_length = 1024` (int64), 1024-sample float32 window, `onesided = 1`.
- `pitch_bin_centers` is a 200-bin **log-spaced** grid from 46.875 Hz to 2093.75 Hz (~33 cents
  per bin); the graph decodes pitch and confidence internally (softmax over bins, peak-mass
  confidence), so the provider gets decoded arrays, not salience logits.
- License: repo `LICENSE` is MIT, Copyright (c) 2025 Lars Nieradzik; `pyproject.toml` declares
  `license = "MIT"` (PEP 639) and `license-files = ["LICENSE"]`; PyPI metadata reports
  `license_expression = MIT`. The weights live in the licensed repository, and the paper
  ([arXiv:2508.18440], author's preprint) names no separate weights terms.
- Training-data provenance (for completeness, **claimed** by the paper): 5-fold group CV over
  NSynth, PTDB-TUG, MIR-1k, MDB-STEM-Synth and the author's synthetic SpeechSynth; held-out
  evaluation on Bach10-mf0-synth, Vocadito and SpeechSynth. That mix includes singing
  (MIR-1k) and expressive solo voice (Vocadito, held out).

## Runtime: does `ort` run the committed ONNX?

**Verified**, three independent legs:

1. **The chosen `ort` links the right runtime.** `ort` v2.0.0-rc.13's `ort-sys` distribution
   table (`ort-sys/build/download/dist.tsv` at tag `v2.0.0-rc.13`) serves every platform archive
   from `ms@1.28.0` - e.g. `x86_64-pc-windows-msvc+directml.tar.lzma2`, `aarch64-apple-darwin
   +coreml.tar.lzma2`, `x86_64-unknown-linux-gnu.tar.lzma2`. So rc.13 = ONNX Runtime 1.28.0,
   matching [#48]'s "ort rc.13 (ONNX Runtime 1.28)".
2. **ORT 1.28.0 has the STFT kernel.** `docs/OperatorKernels.md` at tag `v1.28.0` lists under
   `CPUExecutionProvider`: `STFT` - inputs signal/frame_step/window/frame_length, opset
   **17+**, `T1 = tensor(double), tensor(float)`, `T2 = tensor(int32), tensor(int64)`. The model
   uses float32 tensors and int64 frame_step/frame_length, i.e. a supported combination. All
   other ops in the graph are long-standing CPU kernels.
3. **Direct execution, bit-exact across runtime versions.** Loading the repo model in Python
   ONNX Runtime 1.28.0 and 1.30.0 and running five signals (silence, sine, harmonic stack,
   white noise, harmonic + noise) produces **bit-identical `pitch_hz` and `confidence`
   arrays**. The same holds comparing the repo model against the PyPI wheel model. This is the
   exact library version `ort` rc.13 downloads; only the bindings differ (the Rust path was not
   built here, per the ticket's no-harness scope).

### Contract details that matter for integration (**measured**)

- **Frame count.** For input length `L >= 256` at 16 kHz, the graph emits exactly
  `T = floor(L / 256)` frames. Sweep (repo and wheel model, both runtimes, identical):
  `L=256 -> 1`, `512 -> 2`, `768 -> 3`, `1024 -> 4`, `16000 -> 62`, `32000 -> 125`. There is no
  trailing `+1` frame; the built-in 384-sample symmetric pad absorbs the edges.
- **Short inputs fail.** `L < 256` raises inside ORT's STFT kernel:
  `window_size <= signal_size was false` (`dft.cc:570`), for every `L` tested (1, 50, 100, 255).
  The reference Python wrapper therefore zero-pads any input shorter than 256 samples to 256
  (`SwiftF0.MIN_AUDIO_LENGTH`). The provider must port this pad: a 44.1 kHz wav of 256-705
  samples - well above the #47 short-clip short-circuit threshold - resamples to under 256
  samples at 16 kHz.
- **Native time base.** The wrapper documents frame centers at `k*256 + 127.5` samples
  (`CENTER_OFFSET = (1024-1)/2 - 384`), i.e. a **+7.96875 ms origin offset** on top of the
  16 ms hop, consistent with the graph's symmetric padding geometry. #47 already requires a
  per-model constant frame-origin offset ("the wrapper's to report and the mapper's to apply");
  this is the value SwiftF0's adapter must report.
- **Synthetic smoke behavior** (3 s clips, repo model, ORT 1.28.0; **not** corpus evidence):
  - digital silence: 0/187 voiced at the 0.9 threshold; confidence 0.26-0.38;
  - white noise (amp 0.1): 0/187; confidence <= 0.62;
  - 10-harmonic 220 Hz stack, with or without added noise: 187/187 voiced, confidence >=
    0.998, pitch 220.06 Hz - and the confidence is essentially level-independent across
    amplitudes 0.02-0.9, so harmonic content needs no amplitude normalization;
  - pure 220 Hz sine: sparse and strongly level-dependent (amp 0.05: 246/250 voiced;
    amp 0.3: 4/250; amp 0.9: 2/250) - the confidence surface on pure tones is not the
    harmonic-rich one. This echoes Harvest's documented pure-tone gotcha in kind (see
    [world-integration.md](world-integration.md)), but the direction differs: Harvest misses
    the tone entirely, SwiftF0 mostly rejects it at 0.9.
  - Silence and noise never being voiced is the desired fail-safe for UTAU near-silence; the
    pure-tone sensitivity is a fixture-design warning, not a corpus finding.

## Integration delta vs the RMVPE path (#57)

The provider does not exist yet ([#57] is unstarted; `crates/` has no `kirafrq-ml-provider`),
so this compares against the [#48]/[#57] design, not code.

| Concern | RMVPE path (#57) | SwiftF0 | Delta |
| --- | --- | --- | --- |
| Runtime, session, threading | `ort` rc.13, one Mutex'd session, `inter_op = 1`, `jobs` -> intra-op | identical | none |
| 44.1 -> 16 kHz resample | rubato `Fft`, per #47 | identical | none |
| Input preprocessing | log-mel 128 x `n_fft=1024` x hop 160, 30-8000 Hz, x32 pad, in Rust | none; raw `[1, L]` float32 | **less** code |
| Input padding | mel padded x32 | mandatory zero-pad to >=256 samples | new, ~5 lines |
| Native grid | hop 160 (10 ms), origin as reported | hop 256 (16 ms), origin +127.5 samples | per-model descriptor |
| Frame count | model-specific, mapper owns it | `floor(L16/256)` native frames | per-model descriptor |
| Output decoding | `[T, 360]` salience -> Rust local-avg decode + peak salience | graph emits `pitch_hz` + `confidence` | **less** code |
| Voicing default | peak salience >= 0.03 | confidence > 0.9 (paper: "approximately 90%"; benchmark uses 0.887) | second per-model default |
| Frequency range | ~32.7-1975 Hz | 46.875-2093.75 Hz | both cover 71-800 policy |
| Model resolution | user-supplied `rmvpe.onnx`, exe dir / env var | same mechanism, e.g. `swiftf0.onnx` | registry/file-name entry |
| Model size / terms | 361.7 MB, MIT mirror with provenance caveat; not redistributed | 398 KB, author-committed, MIT; bundling is clean if desired | policy option only |
| CLI/GUI | `Estimator::Rmvpe` in the flat list | one more enum variant + labels | trivial |
| Tests / CI | model-free; no model in CI | model-free + a 398 KB real ONNX fixture is CI-feasible | optional new fixture test |

Net new code for a second model: a raw-audio adapter (input prep, session run, two-output
extraction), a per-model descriptor (filename, native hop, origin offset, confidence default,
output mapping), the short-audio pad, mapper offset handling, and enum/UI wiring. No new
preprocessing, no decoder, no new dependency, and ORT static size is unchanged
([#48] measured +20.4 MiB for the ML build as a whole).

## Public evidence for expressive singing and noisy material

All numbers in this section are **claimed** by the owning source (the SwiftF0 author's own
benchmark repo or his paper unless noted), and none of them is UTAU material.

- **Paper ([arXiv:2508.18440], verified abstract and body).** 95,842 parameters; harmonic-mean
  score 94.07% clean dropping to 91.80% at 10 dB SNR (CHiME-Home noise), >12 pp over CREPE;
  ~42x faster than CREPE on CPU. Notably, the paper's evaluation does **not** include RMVPE,
  and its own discussion warns: "since we train only on voiced frames, the resulting scores
  primarily reflect certainty in pitch class, not voicing decisions" - future work lists
  confidence calibration and training with unvoiced frames.
- **`lars76/pitch-benchmark` (MIT, same author; verified from `benchmark_report.md`).**
  Harmonic-mean average across 8 datasets: **SwiftF0 90.2%, RMVPE 87.2%, CREPE 85.3%**.
  Per dataset (SwiftF0 vs RMVPE): Vocadito **92.6 vs 96.4** (solo expressive singing - RMVPE
  leads), MIR1K 95.0 vs 96.0, PTDBNoisy **74.0 vs 68.5** (real noisy speech - SwiftF0 leads),
  NSynth 89.3 vs 68.2 (instrumental), PTDB 90.4 vs 88.9, Bach10Synth 97.5 vs 98.1,
  MDBStemSynth 92.0 vs 90.6, SpeechSynth 90.7 vs 90.6.
- **Voicing (benchmark, aggregated across datasets).** SwiftF0 F1 0.885 (precision 0.903,
  recall 0.871) vs RMVPE F1 0.837 (precision 0.902, recall 0.793): SwiftF0's edge is recall -
  relevant to UTAU breath where missing voiced frames and voicing noise are both failure modes.
- **Octave stability: sources conflict.** The benchmark's aggregate octave-error rate favors
  SwiftF0 (0.012 vs RMVPE 0.020); the third-party `pitch-core` analysis that [#36] cited reports
  the opposite on MIR-1K/Vocadito (RMVPE 0.0-0.1%, SwiftF0 0.8-3.3%). No source compares them
  on UTAU vibrato.
- **Benchmark caveats (verified from the repo).** The harness normalizes audio to `[-1, 1]`
  before running algorithms, sweeps voicing thresholds 0.0-1.0 and can report the per-algorithm
  optimal threshold (SwiftF0 default 0.887, RMVPE 0.03); its "CPU time" rows (SwiftF0 91.9x
  CREPE, RMVPE 5.1x) are at 22.05 kHz/256-hop on the author's machine and diverge from this
  repo's measurements (SwiftF0 47-56x realtime, RMVPE 22-41x on CPU; see
  [ml-f0-estimators.md](ml-f0-estimators.md)) - treat both as environment-specific.
- **What no public source says.** Nothing measures either model on UTAU voicebank audio:
  breathy frames, plosive transients, mic-pop noise floors, vibrato at UTAU rates, or
  voiced/unvoiced transitions at the 5.805 ms table grid. Nothing compares SwiftF0 to
  Harvest/DIO. No independent (non-author) SwiftF0-vs-RMVPE study was found. The paper itself
  urges downstream-task evaluation rather than metric-only conclusions.

## Verdict on a corpus prototype

**Warranted, bounded, and after the provider lands.** The runtime and license facts are settled
(no risk left there), and the integration delta is small, so the remaining uncertainty is
exactly the one this repo's own experience says matters most for frequency tables: what the
estimator does in the unvoiced/breathy/noisy regions and around vibrato and note transitions on
real voicebank audio. Public benchmarks are speech/singing corpora, and the model's own author
cautions that the confidence score is a pitch-class score, not a voicing score. A prototype
should run through the real `kirafrq-ml-provider` seam once [#57]/[#56] land, and compare
SwiftF0 against the existing WORLD default and RMVPE on: voiced/unvoiced agreement on
breath/plosive/near-silence frames, octave errors on sustained vibrato, short-note onset
behavior under the #47 both-bracket rule (16 ms native frames make the minimum constructible
voiced run ~2 native frames), and confidence-threshold sanity at UTAU recording levels. The
model file can be fetched into scratch; nothing needs to be bundled. If RMVPE integration is
still in flight, this can wait behind it - it is a second-estimator question, not a blocker for
the map's default path.

## Open questions

- UTAU breath/plosive U/V behavior at the 0.9 default (and whether a per-model threshold
  override is needed; [#53] already parameterizes it).
- Half/double-pitch behavior on UTAU vibrato: benchmark and `pitch-core` disagree in sign.
- Level handling: the wrapper does not normalize amplitude and the model is level-insensitive
  on harmonic content in our probe, but UTAU banks vary widely; the prototype should confirm no
  gain normalization is needed.
- Whether the provider should bundle `swiftf0.onnx` (MIT, 398 KB) instead of requiring a
  user-supplied file - a policy call for [#48]/[#49], not a compatibility question.

## Sources

- [lars76/swift-f0][swiftf0-repo] - `LICENSE` (MIT), `pyproject.toml` (SPDX MIT),
  `README.md`, `CHANGELOG.md`, `swift_f0/core.py` (wrapper contract), `swift_f0/model.onnx`
  at commit `64700fce8ef39c2970814bf427ac1d75a2f20d72`; commit history of the model path
  (initial commit 2025-07-08, cleanup 2025-09-02).
- PyPI `swift-f0` 0.1.2 JSON metadata (`license_expression: MIT`) and its bundled `model.onnx`.
- [arXiv:2508.18440][swiftf0-paper] - abstract and HTML full text (training data, confidence
  discussion, SNR results).
- [lars76/pitch-benchmark][pitch-benchmark] - MIT; `benchmark_report.md`, `algorithms/*.py`
  (thresholds, normalization), `datasets/`.
- [pykeio/ort][ort] tag `v2.0.0-rc.13` - `ort-sys/build/download/dist.tsv` (`ms@1.28.0`),
  `ort-sys/build/download/resolve.rs`.
- [microsoft/onnxruntime][ort-repo] tag `v1.28.0` - `docs/OperatorKernels.md` (CPU `STFT`,
  opset 17+).
- Local measurements this session: Python ONNX Runtime 1.28.0 and 1.30.0 CPU on synthetic
  signals; ONNX graph inspection of the repo and wheel models. Scratch venvs and the downloaded
  model copies live in the harness scratchpad, not in this repo.

[swiftf0-repo]: https://github.com/lars76/swift-f0
[swiftf0-paper]: https://arxiv.org/abs/2508.18440
[pitch-benchmark]: https://github.com/lars76/pitch-benchmark
[ort]: https://github.com/pykeio/ort
[ort-repo]: https://github.com/microsoft/onnxruntime
