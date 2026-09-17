//! Shared model and format IO for UTAU frequency tables.
//!
//! [`FrequencyTable`] is the neutral in-memory model every format module reads
//! and writes (ADR `docs/adr/0001-neutral-frequency-table.md`). This crate
//! takes no position on how f0 was estimated and never links WORLD, so the
//! later editor and check/repair milestones can use it on its own.

pub mod audio;
pub mod frq;
pub mod mrq;
pub mod pmk;

/// The neutral frequency table: one hop grid of f0, the optional per-frame
/// amplitude a requested target needs, and the table's key frequency.
///
/// Frame `i` covers `[i*hop, (i+1)*hop)`; see `docs/research/frame-conventions.md`.
#[derive(Debug, Clone, PartialEq)]
pub struct FrequencyTable {
    /// 44100 post-normalization.
    pub sample_rate: u32,
    /// 256 for the UTAU table formats.
    pub hop_samples: u32,
    /// Frame `i` sits at `i*hop/fs`; `0.0` marks an unvoiced frame.
    pub f0_hz: Vec<f64>,
    /// Same length as `f0_hz` when present; raw writer-scale values (frq only).
    pub amplitude: Option<Vec<f64>>,
    /// Mean of this table's voiced frames (frqeditor 5-8-5); `0.0` when nothing is voiced.
    pub key_hz: f64,
}
