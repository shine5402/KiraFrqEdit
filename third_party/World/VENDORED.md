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
`dio.cpp`, `harvest.cpp`, `stonemask.cpp`, `common.cpp`, `fft.cpp`.

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

Re-vendor: extract a fresh upstream `src/` tree, then
`git apply third_party/World/patches/0001-progress.patch` from the repo root
(verified with `git apply --check` against a pristine tree).
