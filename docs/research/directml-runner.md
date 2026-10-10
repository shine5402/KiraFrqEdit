# DirectML GPU runner for the ML estimators

Research for a Windows GPU option for the ML f0 estimators (RMVPE, SwiftF0),
collected 2026-10. Claims are marked **verified** (checked against the pinned
`ort` 2.0.0-rc.13 / ONNX Runtime 1.28 artifacts or upstream source while
writing this) or **cited** (upstream documentation). The "Design" section
records the landed design; the code is the source of truth.

## Summary

- Feasible and mostly free on the `ort` side: the `directml` Cargo feature
  exposes `ep::DirectML`, including per-adapter selection
  (`with_device_id`) and auto-selection (`with_device_filter` /
  `with_performance_preference`).
- The prebuilt Windows ONNX Runtime that `ort` downloads already has the DML EP
  compiled in, and the same archive ships `DirectML.dll` (v1.15.4), which
  `ort-sys`'s `copy-dylibs` seeds into the build's `target/` folder.
- DirectML keeps ONNX Runtime's built-in CPU fallback: operations the DirectML
  EP does not take run on the CPU. A strict mode that disables the fallback
  cannot commit either current model (see Empirical verification), so it is
  not part of the design.
- Device choice: ORT defines `device_id` as the **DXGI adapter index**
  (`IDXGIFactory::EnumAdapters`), with 0 always the default adapter — which on
  laptops is often the iGPU. ORT has no adapter names; a named picker needs
  DXGI enumeration on our side.
- Both models' op sets have DML kernels in ORT 1.28 (STFT 17+, GRU 14+), and
  RVC already runs RMVPE on `DmlExecutionProvider` in the wild, but registered
  kernel ≠ guaranteed placement: the probe below confirms each graph actually
  runs and measures CPU-vs-GPU on a real model.
- Costs: the Windows artifact must ship `DirectML.dll` (~18.5 MB, a load-time
  dependency) with Microsoft's license and third-party notices — even for
  CPU-only users of a DML-enabled build.

## What `ort` 2.0.0-rc.13 provides (verified)

- `directml` feature and `ep::DirectML` exist in the pinned version:
  <https://docs.rs/ort/2.0.0-rc.13/src/ort/ep/directml.rs.html>. The feature is
  Windows-only in practice: `ort-sys`'s download table has a `directml` feature
  set only for `*pc-windows-msvc` targets:
  <https://docs.rs/crate/ort-sys/2.0.0-rc.13/source/build/download/dist.tsv>.
  There is no non-Windows DirectML archive, so only Windows builds enable it.
- Two registration paths (source above):
  - `.with_device_id(i32)` → `SessionOptionsAppendExecutionProvider_DML`
    (legacy; picks the adapter by index). It overrides the filter/preference.
  - no device id → `SessionOptionsAppendExecutionProvider_DML2` with
    `OrtDmlDeviceOptions { Filter: Any|Gpu|Npu, Preference:
    Default|HighPerformance|MinimumPower }` — i.e. ORT can auto-pick a GPU by
    policy. `MinimumPower` is the laptop iGPU-friendly choice.
- Registration failures are **silent by default**: "ort will silently fail and
  fall back to executing on the CPU if all execution providers fail to
  register" (<https://ort.pyke.io/perf/execution-providers>). The shipped
  DirectML runner keeps that behavior (DirectML is a preference, not a
  guarantee); `.error_on_failure()` is the one-line switch if a hard mode is
  ever wanted.
- `SessionBuilder::with_disable_cpu_fallback()` — "Disables falling back to the
  CPU for operations not supported by any other EP. Models with graphs that
  cannot be placed entirely on the EP(s) will fail to commit."
  (<https://docs.rs/ort/2.0.0-rc.13/ort/session/builder/struct.SessionBuilder.html>).
  Not used by the landed design: the probe below shows it cannot build either
  current model.
- `DMLSessionBuilderExt` (same source as the EP) exposes the underlying
  `IDMLDevice` / `ID3D12CommandQueue` after registration. Not needed for v1.
- `ort::device` / `Environment::devices()` exist for ORT's EP-device discovery
  (api-22+), but `HardwareDevice` carries type/vendor/id and **no name**, and
  its `id()` is documented as "not guaranteed to be the same as the device
  *index* for e.g. GPUs"
  (<https://docs.rs/ort/2.0.0-rc.13/src/ort/device.rs.html>). Not a device
  picker for our UI; keep `with_device_id` and enumerate with DXGI.

## ONNX Runtime DML semantics (cited)

- `device_id` is the DXGI adapter index and 0 is the default adapter:
  "The device ID corresponds to the enumeration order of hardware adapters as
  given by IDXGIFactory::EnumAdapters. A device_id of 0 always corresponds to
  the default adapter, which is typically the primary display GPU installed on
  the system. Beware that in systems with multiple GPU's, the primary display
  (GPU 0) is often not the most performant one, particularly on laptops with
  dual adapters…" — <https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html>.
- The DML EP does not support memory-pattern optimization and requires
  sequential execution; ORT logs and disables the memory pattern itself on
  current versions, but `.with_memory_pattern(false)` is the documented
  requirement if a host ever errors. Only one thread may call `Run` per session
  — our process-wide session is already mutex-serialized
  (`crates/kirafrq-ml-provider/src/session.rs`).
- Dynamic input shapes cost: DML "performs best when the size of inputs &
  outputs are known when the session is created". RMVPE's mel input is
  `[1, 128, T]` with `T` varying per wav, so expect per-shape replanning; this
  is a benchmark item, not a blocker.
- The docs banner says DirectML is in **sustained engineering** (supported, new
  feature work has moved to WinML) and that ORT uses DirectML 1.15.2. `ort`
  still ships first-class bindings, and our pinned archive actually bundles
  DirectML 1.15.4 (below).

## The Windows binaries and `DirectML.dll` (verified)

- `dist.tsv` (pinned rc.13, ORT 1.28.0) lists the DML archive
  `x86_64-pc-windows-msvc+directml.tar.lzma2`; it contains exactly two members,
  `onnxruntime.lib` and `DirectML.dll` (18,527,776 bytes, "DirectML
  Redistributable Library" 1.15.4). Hashes and provenance live in
  `third_party/DirectML/VENDORED.md`.
- `ort-sys`'s `copy-dylibs` copies `.dll`/`.so`/`.dylib` files from the
  extracted archive into the Cargo target dirs at build time
  (<https://docs.rs/crate/ort-sys/2.0.0-rc.13/source/build/dynamic_link.rs>).
  On Windows without Developer Mode it copies into the profile root only (e.g.
  `target/release/`), so test/example binaries under `deps/` rely on the
  machine's `System32` copy.
- Downstream evidence that the built artifact needs the DLL at **load** time:
  <https://github.com/xberg-io/xberg/issues/1456> — "Windows builds link
  DirectML.dll … They do not ship that DLL. Loading fails on every host that
  does not already provide it", including the fix (ship it next to the
  artifact). Desktop Windows 10/11 ship `DirectML.dll` in `System32`; Server
  Core and minimal images do not.
- Consequence for CD: add `target/release/DirectML.dll` to the Windows
  artifact. For CPU-only builds the existing `--no-default-features` (compat)
  artifact is unaffected because it links no ORT.
- License: the `Microsoft.AI.DirectML` **1.15.4** package ships `LICENSE.txt` =
  Microsoft DirectML license terms, §1(a): "you may install and use any number
  of copies of the software, and solely for use on Windows and Xbox. You may
  copy and distribute the software … in applications and services you
  develop…"; the package also has `ThirdPartyNotices.txt` (GSL, half, zstd)
  that must travel with redistributions, and an MIT `LICENSE-CODE.txt` for the
  package code. Redistribution in our binaries is fine; `CREDITS.md` carries
  the notice and `third_party/DirectML/` the texts (the `about.toml` pipeline
  only covers Rust crates).

## Operator coverage for our two models (verified)

From the DML table in ORT v1.28.0 `OperatorKernels.md`
(<https://github.com/microsoft/onnxruntime/blob/v1.28.0/docs/OperatorKernels.md>,
section "Operators implemented by DmlExecutionProvider"):

| Op | DML kernel |
| --- | --- |
| `STFT` | 17+, f32/f16 — SwiftF0's frontend |
| `GRU` / `LSTM` | 14+, f32/f16 — RMVPE's recurrent layers |
| `Conv`, `ConvTranspose` (dynamic pads), `MatMul`, `Softmax`, `Sigmoid`, `Tanh`, `Erf`, `ReduceMean` | covered |
| `NonZero`, `Range`, `ScatterND`, `Trilu`, `Resize`, `Pad`, `Slice`, `Gather`, `Shape`, `Cast`, control-flow-ish helpers | covered |

Registered kernel ≠ placement guarantee: DML partitions the graph at runtime,
and the probe below is what actually confirmed the two models run.
RMVPE-on-DirectML has a strong in-the-wild precedent: RVC creates
its ONNX RMVPE session with `providers=["DmlExecutionProvider"]`
(<https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/infer/rmvpe.py>,
around the `privateuseone` branch).

## Empirical verification (local Windows machine, 2026-10)

The `directml_probe` example added alongside this note loaded the real RMVPE
model (362 MB, local scratch) and the bundled SwiftF0 over the same synthetic
audio under the CPU and DirectML runners (plus the strict mode during the
probe, before it was dropped), on an integrated DirectX 12 GPU:

| Model | Runner | Session | Output vs CPU | Inference time |
| --- | --- | --- | --- | --- |
| RMVPE | CPU | ok | reference | ~573 ms |
| RMVPE | DirectML | ok | max abs diff 6e-6 Hz | ~362 ms steady, 1.5 s first |
| RMVPE | DirectML (strict, dropped) | **commit fails** | — | — |
| SwiftF0 | CPU | ok | reference | ~46 ms |
| SwiftF0 | DirectML | ok | max abs diff 3e-5 Hz | ~4.3 ms steady, 229 ms first |
| SwiftF0 | DirectML (strict, dropped) | **commit fails** | — | — |

- DirectML works for both models, and the policy-applied f0 matches the CPU
  path to f32 kernel noise. The speedup on an integrated GPU is already ~1.6x
  (RMVPE) and ~10x (SwiftF0); a discrete GPU should widen it.
- The experimental strict mode (`session.disable_cpu_ep_fallback`) failed for
  both models with ORT's "This session contains graph nodes that are assigned
  to the default CPU EP, but fallback to CPU EP has been explicitly disabled
  by the user." The DML kernel table has `Shape`/`Gather`/`Reshape`/`Where`
  etc., so this is the partitioner leaving shape-handling nodes on the CPU
  under the models' dynamic input axes, not a missing kernel — the mode is
  unusable for these graphs and was dropped.
- Timings are single-machine with a fixed 3 s input; first inference includes
  DML graph initialization. RMVPE's mel frames vary per wav, so a batch may
  add per-shape graph work the probe does not show.

## How other tools expose it (verified)

- **VC Client / w-okada voice-changer** (same domain: RVC + ONNX f0 + DML):
  after <https://github.com/w-okada/voice-changer/issues/410> it shipped
  `GPU0` / `GPU1` buttons — raw indices, not names; the maintainer: "select id
  will be in next release (not device name), only for experimental". The thread
  is a UX cautionary tale: the very next question was "How do I know which ID
  corresponds to which GPU?", answered with a DXGI `EnumAdapters` snippet and,
  for Python, `torch_directml.device_name(i)`. Later users were still confused.
- **RVC WebUI**: runs RMVPE on the default DML adapter
  (`providers=["DmlExecutionProvider"]`, no device option); the torch-directml
  device is chosen elsewhere.
- **stable-diffusion-webui-amdgpu** (DirectML fork): exposes `--device-id N`
  only, no enumeration; users are told to try indices
  (<https://github.com/lshqqytiger/stable-diffusion-webui-amdgpu/discussions/149>).

Synthesis: nobody in this space ships a *named* adapter picker; the index-only
approach demonstrably confuses users, so a DXGI-named list is a differentiator,
and "Automatic" should stay the default (it is also the only sane choice when
enumeration fails).

## Design (as landed)

- **Config surface** (`kirafrqgen-core`): extend `MlConfig` with
  `runner: MlRunner` (`Cpu` default; `DirectMl`) and
  `directml_device: Option<u32>` (`None` = automatic). WORLD estimators ignore
  both. `kirafrq-ml-provider` maps the neutral choice onto its session builder
  (it must keep not depending on core), and `session.rs`'s cache key gains
  `(runner, device)` so switching rebuilds the singleton.
- **EP wiring**: only for `Rmvpe`/`SwiftF0` sessions. The DirectML runner
  registers `ep::DirectML::default().with_device_id(n)` (explicit) or the DML2
  path (`None` = `PerformancePreference::HighPerformance`). ONNX Runtime's
  built-in CPU fallback stays on, matching DirectML's design.
- **GUI**: hamburger (`icons::LIST`) → "Advanced…" opens a window next to the
  Credits one: ML runner radios (CPU / DirectML) and a "DirectML device"
  combo — "Automatic" plus `N — <DXGI description>` entries (mark index 0 as
  default), disabled unless DirectML is selected and disabled while a run is
  active. Windows-only; hidden when the build has no DML. Session-only
  persistence, matching #13, with a Refresh devices button.
- **Device enumeration** (Windows, `cfg(windows)`): the `windows` crate
  (`Win32_Graphics_Dxgi`) walks `IDXGIFactory1::EnumAdapters1` →
  `DXGI_ADAPTER_DESC1.Description`; the index is the DML `device_id`. Software
  adapters (WARP) are excluded. If enumeration fails, the UI offers
  "Automatic" and shows the reason.
- **CLI parity**: `--ml-runner cpu|directml`, `--ml-device N` (only with a
  DirectML runner), `--list-ml-devices`; CPU is the default.
- **Build gating**: a Windows-only target dependency in
  `kirafrq-ml-provider`'s manifest enables `ort/directml`; non-Windows builds
  and the GUI compile unchanged.
- **CD**: the release workflow copies `target/release/DirectML.dll` and the
  texts from `third_party/DirectML/` beside the Windows binaries;
  `CREDITS.md` carries the notice.

## Remaining verification

1. Benchmark a batch CPU vs DML, including a mixed-length set to expose the
   per-shape replanning cost; try `.with_memory_pattern(false)` if needed. A
   profiling follow-up could name the exact CPU-assigned nodes.
2. CI: compile-only for `directml`; no DML sessions in unit tests (runners have
   no GPU). Keep session-free unit tests for runner mapping and device
   formatting.
3. Later, a Windows artifact smoke test on a minimal image (no `DirectML.dll`
   in `System32`) to prove the bundled DLL is found — the xberg issue shows a
   normal CI image cannot catch that.

## Open questions

- Automatic device semantics: the design uses DML2 `HighPerformance`; should
  `MinimumPower` (or the choice) be exposed?

## Sources

- ort execution providers: <https://ort.pyke.io/perf/execution-providers>
- ort prebuilt binaries (Windows builds carry DML): <https://ort.pyke.io/misc/prebuilt-binaries>
- ort rc.13 DirectML source (`with_device_id`, DML2, `DMLSessionBuilderExt`): <https://docs.rs/ort/2.0.0-rc.13/src/ort/ep/directml.rs.html>
- ort rc.13 `SessionBuilder` (`with_disable_cpu_fallback`, `with_memory_pattern`): <https://docs.rs/ort/2.0.0-rc.13/ort/session/builder/struct.SessionBuilder.html>
- ort rc.13 device discovery (`id()` caveat): <https://docs.rs/ort/2.0.0-rc.13/src/ort/device.rs.html>
- ort-sys dist table: <https://docs.rs/crate/ort-sys/2.0.0-rc.13/source/build/download/dist.tsv>
- ort-sys `copy-dylibs`: <https://docs.rs/crate/ort-sys/2.0.0-rc.13/source/build/dynamic_link.rs>
- ONNX Runtime DirectML EP (device_id = DXGI index, constraints, opset 20, sustained engineering): <https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html>
- ONNX Runtime session config key `session.disable_cpu_ep_fallback`: <https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h>
- ORT v1.28.0 operator kernels (DML table): <https://github.com/microsoft/onnxruntime/blob/v1.28.0/docs/OperatorKernels.md>
- DirectML.dll load-time dependency, downstream report: <https://github.com/xberg-io/xberg/issues/1456>
- DirectML redistributable license: `Microsoft.AI.DirectML` 1.15.4 package (`LICENSE.txt`, `ThirdPartyNotices.txt`), <https://www.nuget.org/packages/Microsoft.AI.DirectML/1.15.4>
- VC Client GPU-selection thread (index UX, DXGI mapping): <https://github.com/w-okada/voice-changer/issues/410>
- RVC RMVPE on DML: <https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/infer/rmvpe.py>
- sd-webui-amdgpu `--device-id`: <https://github.com/lshqqytiger/stable-diffusion-webui-amdgpu/discussions/149>
