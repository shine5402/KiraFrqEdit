# WORLD (vendored)

Upstream: <https://github.com/mmorise/World>

- Version: v1.0.1
- Commit: `d625e7608ca23a870018f01e7c562ac683d9847f` (tag `v1.0.1`, 2025-02-21)
- Retrieved from: `https://codeload.github.com/mmorise/World/tar.gz/refs/tags/v1.0.1` (2026-09-17)
- Contents: upstream `LICENSE.txt` and `src/` tree, plus the instrumentation
  patch below (re-apply with `git apply` after re-vendoring).

License: modified BSD (3-clause); see `LICENSE.txt`. The copyright notice and license text must
be retained in source and binary distributions.

Built by `crates/kirafrq-world-binding/build.rs` (cc crate) from these translation units:
`dio.cpp`, `harvest.cpp`, `stonemask.cpp`, `d4c.cpp`, `common.cpp`, `fft.cpp`,
`matlabfunctions.cpp`.

## Instrumentation patch (#34)

`patches/0001-progress.patch` adds a frame-progress hook call to three
per-frame loops: `GetBestF0Contour` (`dio.cpp`), the candidate-pass loop in
`GetRawF0Candidates` (`harvest.cpp`), and the `StoneMask` loop
(`stonemask.cpp`). Each hunk is an increment plus a call only — no arithmetic,
no branch or loop-bound changes, no reordering — so numerical results are
unchanged. The hook itself (declared in
`crates/kirafrq-world-binding/shim/progress_hook.h`, defined in
`shim/world_shim.cpp`) lives outside the vendored tree and is not part of the
patch.

Note the Harvest site: the issue pointed at the inner frame loop, but that
loop fires once per channel with per-pass resets, which cannot compose into a
monotonic file fraction. The recorded hunk reports each candidate pass as one
step (`done = channel + 1, total = number_of_channels`) instead.

## Aperiodicity-statistic patch (#55)

`patches/0002-d4c-aperiodicity0.patch` exposes the raw D4C LoveTrain
statistic (`D4CLoveTrain`'s `aperiodicity0`) as a public
`GetD4CAperiodicity0()` — the ratio D4C itself compares against
`D4COption::threshold` (default 0.85) to label a frame unvoiced. Without this,
the statistic is computed inside `D4C()` and discarded; the returned
`aperiodicity[0]` is only the binary outcome. Pure addition: a declaration in
`world/d4c.h`, a wrapper after the anonymous namespace in `d4c.cpp` that
reseeds the D4C RNG and forwards to the existing static — no change to `D4C()`
or any arithmetic. Built so `kirafrq-world-binding` can expose
`d4c_aperiodicity0()` for the #55 gate probe.

Re-vendor: extract a fresh upstream `src/` tree, then
`git apply third_party/World/patches/0001-progress.patch` and
`git apply third_party/World/patches/0002-d4c-aperiodicity0.patch` from the
repo root (both verified with `git apply --check` against a pristine tree).
