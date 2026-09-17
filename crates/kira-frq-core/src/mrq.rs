//! moresampler `desc.mrq`: per-folder f0 store.
//!
//! Format spec: `docs/research/mrq-format.md`. Merge and staleness policy was
//! settled in #10; the writer and `.llsm` invalidation land via #16. `Desc`
//! survives as a native type because merge must keep untouched entries
//! verbatim (ADR 0001).
