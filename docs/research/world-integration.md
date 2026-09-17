# WORLD integration spike: vendored build and Rust wrapper

Status: task spike for wayfinder ticket [#6] on branch `research/world-integration`
(2026-09-17). Proves the vendored-WORLD build path on Windows with both MSVC and clang-cl, and
the crate/module layout that will graduate into the workspace scaffold. Cross-platform is a
standing goal: the build stays toolchain-agnostic, with Unix clang/gcc expected to work (not
exercised on this machine; see below). Feeds [#7] (core analysis-to-writer API).

[#6]: https://github.com/shine5402/KiraFrqEdit/issues/6
[#7]: https://github.com/shine5402/KiraFrqEdit/issues/7

## Vendored source

- Upstream: <https://github.com/mmorise/World>, tag `v1.0.1`, commit
  `d625e7608ca23a870018f01e7c562ac683d9847f` (2025-02-21).
- Retrieved 2026-09-17 from `https://codeload.github.com/mmorise/World/tar.gz/refs/tags/v1.0.1`.
- Committed unmodified at `third_party/World/` (`LICENSE.txt` + full upstream `src/`); pin and
  provenance in `third_party/World/VENDORED.md`. No submodule.

## Build mechanics (`cc` 1.4.6; MSVC and clang-cl)

`crates/frq-world/build.rs` compiles six translation units into static lib `world`:
`dio.cpp`, `harvest.cpp`, `stonemask.cpp`, `common.cpp`, `fft.cpp`, **`matlabfunctions.cpp`**.

- `matlabfunctions.cpp` is required, despite the name: `common.cpp` needs `interp1Q` and
  `dio.cpp`/`harvest.cpp`/`stonemask.cpp` need `interp1`, `decimate`, `matlab_round`. Omitting it
  fails at link time (observed: `LNK2019` unresolved `interp1`, `decimate`, `matlab_round`,
  `interp1Q`). The upstream CMake target includes it too.
- Build config: `cpp(true)`, `include("third_party/World/src")`, `warnings(false)`,
  `flag_if_supported("/EHsc")`, default MSVC C++ standard (C++14).
- `warnings(false)` is deliberate: MSVC emits only `C4100` unreferenced-parameter warnings in
  upstream `dio.cpp` (`fs`) and `harvest.cpp` (`actual_fs`, `vuv`, `f0_length`, `f0`). The shim
  and wrapper compile warning-free; we do not patch the vendored tree.
- Toolchain exercised: rustc/cargo 1.98.1 (`x86_64-pc-windows-msvc`), VS 2022 Build Tools
  (`cl.exe` 14.44.35207), `cc` 1.4.6, `hound` 3.5.1. `cargo test` needs no vcvars shell; `cc`
  locates MSVC itself.
- Same tree builds with LLVM on Windows: `CC=CXX="clang-cl"` (LLVM on PATH; clang-cl 23.1.1),
  `cargo clean` first so `cc` re-runs. All tests pass and f0 output is
  identical to the MSVC build (voiced counts and means match exactly). `cc` picks the archiver;
  no extra configuration.
- `build.rs` is toolchain-agnostic: `/EHsc` goes through `flag_if_supported`, so GNU-style
  clang/gcc would skip it, and upstream WORLD builds with gcc/clang via its own CMake/makefile.
  Unix toolchains are expected to work but were not exercised on this machine; the CI matrix
  belongs to the workspace-scaffold ticket.

## Shim style: raw `extern "C"` shim (cxx not needed)

- WORLD's public functions are already `extern "C"` (`WORLD_BEGIN_C_DECLS` in
  `world/macrodefinitions.h`), but its option structs (`DioOption`, `HarvestOption`) live in
  headers. The spike took the thinnest path: `shim/world_shim.{h,cpp}` (six functions taking
  scalars/arrays) builds and links fine; option structs are initialized C++-side and only
  `f0_floor`/`f0_ceil`/`frame_period` are overridden, so Rust never mirrors a C++ layout.
- Rust side: `src/sys.rs` holds every `unsafe extern "C"` declaration and the only `unsafe`
  blocks; `src/lib.rs` is fully safe. Crate uses no runtime dependencies.
- Rejected: `cxx` (no shared types cross the boundary, so its codegen and dependency buy
  nothing), bindgen + `#[repr(C)]` option structs (duplicates upstream layout, couples the
  wrapper to WORLD struct changes).
- `x_length` is a C `int`: the wrapper returns `WorldError::TooLong` above `i32::MAX` samples
  (~13.5 h at 44.1 kHz) instead of truncating.

## API surface used

| Call | Role |
| --- | --- |
| `InitializeDioOption` / `Dio` / `GetSamplesForDIO` | DIO f0; defaults kept: `channels_in_octave` 2.0, `speed` 1, `allowed_range` 0.1 |
| `InitializeHarvestOption` / `Harvest` / `GetSamplesForHarvest` | Harvest f0; defaults kept; internals: 8 kHz target, 40 channels/octave, 1 ms internal step resampled to `frame_period` |
| `StoneMask` | refinement of either estimator's f0; takes no options |

`KiraFrqGen` defaults applied on top: floor 71 Hz, ceiling 800 Hz, frame period 5 ms
(`frq_world::F0Options`).

## Verified behavior and performance

- Frame count is `floor(1000 * x_length / fs / frame_period) + 1` for both estimators, and
  `temporal_positions[n] = n * frame_period / 1000` exactly (5.000 ms spacing asserted).
- Real voicebank wav (a 5.48 s, 44.1 kHz 16-bit mono corpus sample):
  DIO 679/1097 frames voiced, mean 111.24 Hz; Harvest 719/1097, mean 111.38 Hz; StoneMask moves
  the mean by <0.1 Hz. The sibling `_wav.frq` (945 frames, existing engine output) has mean
  116.66 Hz, min 55.38, max 858.57 - same ballpark, but different frame grid and a wider
  floor/ceiling than the 71/800 policy, so no frame-level comparison was attempted.
- Release timings (no LTO), Windows MSVC, this machine:

  | Input | Estimator | Time | Realtime | +StoneMask |
  | --- | --- | --- | --- | --- |
  | synthetic 4.0 s harmonic sweep | DIO | 82 ms | 49x | +47 ms |
  | synthetic 4.0 s harmonic sweep | Harvest | 466 ms | 9x | +28 ms |
  | real 5.48 s voice wav | DIO | 97 ms | 56x | +124 ms |
  | real 5.48 s voice wav | Harvest | 1209 ms | 5x | +63 ms |

  clang-cl 23.1.1 on the same inputs is slightly faster: synthetic sweep DIO 75 ms (53x) / +49 ms;
  Harvest 441 ms (9x) / +25 ms; real wav DIO 83 ms (66x) / +111 ms; Harvest 1166 ms (5x) / +62 ms.
  Debug builds are roughly 2-3x slower. Harvest costs about one order of magnitude more than DIO,
  as expected.
- Tests (`crates/frq-world/tests/estimate.rs`): 5 ms grid + frame count, both estimators
  within 1% of a 220 Hz harmonic tone, floor/ceiling actually moving the voiced set (60 Hz tone
  with 71 vs 50 Hz floor), digital silence unvoiced, input validation. Reproduce:
  `cargo test --manifest-path crates/frq-world/Cargo.toml`.
- Example (`cargo run --release --example world_spike [-- --floor N --ceiling N --seconds N]
  [--wav <path>]`) writes the input wav and per-frame f0 CSVs under `.local/world-spike/`
  (gitignored).

## Gotcha: Harvest needs a spectrally rich signal

Harvest's voicing decision collapses on a pure single sinusoid; DIO is unaffected. Measured on
3 s 220 Hz variants (voiced frames out of 601):

| Signal | DIO | Harvest |
| --- | --- | --- |
| clean sine | 600 | 1 |
| sine + 1% noise | 600 | 601 |
| AM sine (4 Hz) | 600 | 1 |
| 10-harmonic stack | 600 | 601 |
| harmonic stack + 1% noise | 600 | 601 |

Real voicebank audio is unaffected (see above). Synthetic tests/benchmarks must use harmonic-rich
or voiced-with-noise signals; a pure-tone fixture will make Harvest look broken when it is not.

## License obligations

WORLD is modified BSD (3-clause). `third_party/World/LICENSE.txt` is committed; source and binary
distributions must retain the copyright notice, the conditions and the disclaimer. The crate
declares `license = "BSD-3-Clause"`. Upstream README states no patents are claimed.

## Layout on this branch

```
third_party/World/          vendored upstream (LICENSE.txt, VENDORED.md, src/)
crates/frq-world/
  build.rs                  cc build of the six TUs plus the shim
  shim/world_shim.{h,cpp}   thin C ABI; option structs stay C++-side
  src/sys.rs                the only module with unsafe
  src/lib.rs                safe API: Estimator, F0Options, F0Track, WorldError
  examples/world_spike.rs   synthetic sweep or --wav, timings, f0 CSV
  tests/estimate.rs
```

Crate path is `crates/frq-world` pending the workspace-scaffold ticket; the crate is
standalone (no workspace membership yet) and `Cargo.lock` is committed for reproducibility.
