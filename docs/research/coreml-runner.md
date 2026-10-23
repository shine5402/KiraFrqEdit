# Core ML runner for the ML estimators

The Core ML ML runner (#84, spec #83) runs RMVPE and SwiftF0 through ONNX Runtime's Core ML
execution provider on Apple-silicon Macs. This note records why it is configured the way it is,
what was measured, and the ONNX Runtime caveats that bound it. Code: `kirafrq-ml-provider`'s
`coreml` module; the local probe is the `runner_probe` example.

## Summary

- ONNX Runtime's Core ML execution provider with the **NeuralNetwork** model format,
  **CPU-and-GPU** compute units, dynamic input shapes and low-precision GPU accumulation off.
- On an M4 Max, RMVPE runs at ~10 ms per audio-second warm against ~18–19 ms on CPU (~1.8×),
  with ≤0.46 cents of pitch drift (RMVPE bins are 20 cents) and no voiced/unvoiced flips.
- The Core ML compile is cached on disk under `~/Library/Caches/KiraFrqGen/coreml/`, one
  subdirectory per model content, ONNX Runtime build and compute units.
- SwiftF0 gains nothing: Core ML takes only 8–16 of its 51 nodes and the STFT stays on CPU.
  It runs correctly under Core ML, which is all the runner needs from it.
- A native Core ML model (RMVPE converted with coremltools) was spiked and rejected: slower
  warm, no escape from the per-length stall, and the Neural Engine unusable.

## Build gating

- ONNX Runtime 1.28.0's prebuilt archive for `aarch64-apple-darwin` is the only Core ML build
  `ort-sys` 2.0.0-rc.13 downloads (its `dist.tsv` has `aarch64-apple-darwin coreml`), and there is
  no `x86_64-apple-darwin` archive at all. `kirafrq-ml-provider` enables `ort/coreml` for macOS
  targets, the way `ort/directml` is enabled for Windows.
- `COREML_SUPPORTED` is true only for Apple-silicon macOS `ort` builds. Elsewhere the runner
  stays selectable in the CLI and fails the session with "Core ML is only available in
  Apple-silicon macOS ML builds", a configuration error before any file runs.
- The execution provider registers with `error_on_failure`: a Core ML that cannot register
  fails the session instead of quietly running on the CPU, so a committed session really runs
  Core ML. Node-level fallback is separate and stays on: ONNX Runtime's implicit CPU execution
  provider takes the nodes Core ML does not (for RMVPE, the BiGRU and a few shape ops: Core ML
  takes 326 of 339 nodes).
- The capability-aware default is Core ML when a probe session over the bundled SwiftF0
  commits, else CPU. The probe runs once per process and costs ~10–20 ms with a warm cache.

## Execution-provider options (M4 Max, #83)

Measured with RMVPE on synthetic voiced audio, 0.9–5.2 s clips.

| Option | Choice | Why |
| --- | --- | --- |
| Model format | NeuralNetwork | MLProgram re-specializes for every new input shape: ~250–400 ms per new wav length, cached only for the shape seen last. NeuralNetwork costs ~5–10 ms per new length, so a voicebank of differently sized wavs stays fast. |
| Compute units | CPU and GPU | ALL measured the same speed and output, and lets the Neural Engine in, which loses accuracy on quiet audio-model inputs (ONNX Runtime #33319). |
| Input shapes | dynamic | Every wav has its own length. |
| Low-precision GPU accumulation | off | Full-precision accumulation keeps the drift against CPU under half a cent. |
| Model cache directory | on | Later starts skip the Core ML compile: 2.16 s → 0.15 s in #83, before the content hash existed. |

### Parallel sessions

One process can hold several sessions of the same model, each with its own ~360 MB of RMVPE
weights. Realtime factor by session count (#83):

| Sessions | Core ML | CPU |
| --- | --- | --- |
| 1 | 75× | 50× |
| 2 | 138× | 89× |
| 4 | 226× | 124× |

Local runs keep one session; the ML server (#85) is where more sessions pay off.

### The model cache

- ONNX Runtime keys the Core ML compile inside the cache directory by a hash of the model path
  (or a `CACHE_KEY` metadata entry) and never invalidates it, so a model replaced at the same path
  would reuse a stale compile. Each model therefore gets its own subdirectory:
  `coreml/<model SHA-256>/ort-<hash of ort::info()>-CPUAndGPU`. The SHA-256 is of the model's
  current bytes; `ort::info()` names the ONNX Runtime release and commit.
- Hashing RMVPE's 362 MB takes ~0.16 s with `sha2` 0.11's aarch64 SHA instructions (0.10 needed
  its `asm` feature for those and took ~0.68 s in software).
- The cache holds ~669 MB for RMVPE and ~96 KB for SwiftF0. Nothing prunes the subdirectories
  of replaced models or older ONNX Runtime builds; deleting `~/Library/Caches/KiraFrqGen/coreml`
  is always safe.

## Local reproduction (M4 Max, macOS 27.0, 2026-10)

`runner_probe` with the real RMVPE over four synthetic gliding harmonic stacks (0.9, 2.1, 3.4 and
5.2 s), release build, `jobs = 0` (ONNX Runtime's default threads):

| | CPU | Core ML |
| --- | --- | --- |
| Session build, cold cache | 68–82 ms | 1.76 s |
| Session build, warm cache | — | ~0.33 s, including the ~0.16 s hash |
| Warm, per audio-second | 17.8–17.9 ms | 10.1–10.2 ms |
| First call at a new length, minus warm | ~0 | ~3–7 ms; the first call of all ~30 ms |
| Max pitch drift vs CPU | reference | 0.066 Hz (0.45 cents), 0 of 1,164 frames flip |

The options, varied one at a time against the shipped NeuralNetwork + CPU-and-GPU (a separate
cache per run, so a cold and a warm-cache session each; times from the warm-cache run):

| Format, compute units | Warm, per audio-second | First call at a new length | Drift vs CPU |
| --- | --- | --- | --- |
| NeuralNetwork, CPU and GPU (shipped) | 10.1–10.3 ms | ~3–7 ms over warm | 0.45 cents |
| NeuralNetwork, ALL | 10.4 ms | the first call of all ~165 ms, then as above | 0.45 cents |
| MLProgram, CPU and GPU | 35.8 ms* | ~260 ms at every length | 0.00 cents |
| MLProgram, ALL | 37.6 ms* | ~255–285 ms at every length; the first ~675 ms | 0.00 cents |

\* MLProgram keeps only the shape seen last, so the probe's warm loop, which steps through four
lengths, pays one ~260 ms re-specialization per length. A real voicebank changes length on
nearly every wav, so this mixed figure, not the single-length speed, is what a run sees.

SwiftF0 under Core ML matched CPU to 1.5e-5 Hz with no flips, at the same ~2.5–4 ms per
audio-second as CPU. The provider test `tests/coreml.rs` pins the SwiftF0 parity and the cache
keying on Apple-silicon CI.

Run it with
`cargo run --release -p kirafrq-ml-provider --features ort --example runner_probe -- --rmvpe <rmvpe.onnx> [--wav <file.wav>]`.

## Native Core ML model: spiked and rejected (#83)

RMVPE's ONNX graph was rebuilt in PyTorch (matching ONNX Runtime to 4e-7) and converted with
coremltools to MLProgram in four variants: a length range or 100 enumerated lengths, each in fp32
and fp16. Each ran on CPU-and-GPU, ALL and CPU-and-Neural-Engine over 16 real-frontend inputs.

- **Slower warm.** The best was 19–22 ms per audio-second: about 2× slower than ONNX Runtime's
  Core ML path and slower than ONNX Runtime on CPU (~14 ms in that run). The likely cause is the
  BiGRU's sequential recurrence running on the GPU; ONNX Runtime keeps it on the CPU.
- **No escape from the per-length stall.** A new input length still cost a median 85–100 ms (max
  ~300 ms), with or without enumerated shapes.
- **The Neural Engine is unusable for this model.** fp32 with ALL failed with a Core ML error
  ("E5RT: Port descriptor cannot be changed while memory object is bound to port"); fp32 with
  CPU-and-Neural-Engine ran at CPU speed, because fp32 is barred from the Neural Engine; fp16
  with ALL hung inside one prediction for 40 minutes; fp16 (range) and fp32 (enumerated) on
  CPU-and-Neural-Engine produced no result within 5 minutes.
- **No startup gain.** 1–8 s, against 0.15 s for ONNX Runtime with its model cache (#83's figure,
  without the content hash).
- **Accuracy was fine:** fp32 matched exactly; fp16 drifted ≤0.47 cents with no flips.

## ONNX Runtime caveats

- **The 1.29 slowdown.** ONNX Runtime #32218 reports a ~10× Core ML slowdown on iOS in 1.29
  against 1.28; re-run `runner_probe` on macOS before moving the `ort` pin past 1.28.
- **Neural Engine accuracy.** ONNX Runtime #33319: CPU-and-Neural-Engine loses accuracy on quiet
  audio-model log-power inputs. This is why the compute units exclude it.
- **Not actively developed.** A maintainer comment on ONNX Runtime #26328 says the Core ML
  execution provider is not being actively developed upstream, so CPU stays the reliable fallback
  and the default only picks Core ML after a session commits.

## Sources

- ort rc.13 Core ML options (`ModelFormat`, `ComputeUnits`, `with_model_cache_dir` and its
  cache-key note): <https://docs.rs/ort/2.0.0-rc.13/src/ort/ep/coreml.rs.html>
- ort-sys rc.13 dist table: <https://docs.rs/crate/ort-sys/2.0.0-rc.13/source/build/download/dist.tsv>
- ONNX Runtime Core ML execution provider: <https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html>
- ONNX Runtime #33319, Neural Engine accuracy loss on quiet audio: <https://github.com/microsoft/onnxruntime/issues/33319>
- ONNX Runtime #32218, Core ML slowdown in 1.29: <https://github.com/microsoft/onnxruntime/issues/32218>
- ONNX Runtime #26328, Core ML reshape with dynamic shapes (maintainer comment on development
  status): <https://github.com/microsoft/onnxruntime/issues/26328>
- The benchmark and spike: spec issue #83.
