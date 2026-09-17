//! The KiraFrqGen generation pipeline: scan a voicebank for wavs, estimate f0
//! through the [`F0Estimator`] seam, and write frequency tables via
//! frq-core.
//!
//! [`generate`] implements #11's input policy, #8's table policy and the batch
//! semantics of #12; [`generate_wavs`] runs the same pass over an explicit wav
//! list; [`plan`] is the dry-run path (scan plus existence checks, no decode
//! or analysis).

pub mod llsm;

mod paths;
mod run;
mod scan;
mod table;

pub use frq_core::mrq::Sharing;
pub use run::{generate, generate_wavs, plan};

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

/// f0 estimator selection for [`F0Config`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Estimator {
    Dio,
    Harvest,
}

/// f0 estimation settings shared by every wav in a run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct F0Config {
    pub estimator: Estimator,
    pub floor_hz: f64,
    pub ceiling_hz: f64,
    pub stone_mask: bool,
}

impl Default for F0Config {
    fn default() -> Self {
        // Standing defaults (map Notes): Harvest with StoneMask, 71-800 Hz.
        Self {
            estimator: Estimator::Harvest,
            floor_hz: 71.0,
            ceiling_hz: 800.0,
            stone_mask: true,
        }
    }
}

/// One estimator's f0 output on its own time grid; `0.0` marks unvoiced
/// frames.
#[derive(Debug, Clone, PartialEq)]
pub struct F0Track {
    pub frame_period_ms: f64,
    pub temporal_positions: Vec<f64>,
    pub f0_hz: Vec<f64>,
}

/// The estimator seam: generation talks to f0 estimation only through this
/// trait, so tests inject fakes and the pipeline never names WORLD.
pub trait F0Estimator: Send + Sync {
    fn estimate(
        &self,
        samples: &[f64],
        sample_rate: u32,
        frame_period_ms: f64,
    ) -> Result<F0Track, GeneratorError>;

    /// StoneMask refinement; estimators without it keep the default no-op.
    fn refine_stonemask(
        &self,
        _samples: &[f64],
        _sample_rate: u32,
        _track: &mut F0Track,
    ) -> Result<(), GeneratorError> {
        Ok(())
    }
}

/// [`F0Estimator`] backed by frq-world (DIO / Harvest / StoneMask).
pub struct WorldEstimator {
    config: F0Config,
}

impl WorldEstimator {
    pub fn new(config: F0Config) -> Self {
        Self { config }
    }
}

impl F0Estimator for WorldEstimator {
    fn estimate(
        &self,
        samples: &[f64],
        sample_rate: u32,
        frame_period_ms: f64,
    ) -> Result<F0Track, GeneratorError> {
        let estimator = match self.config.estimator {
            Estimator::Dio => frq_world::Estimator::Dio,
            Estimator::Harvest => frq_world::Estimator::Harvest,
        };
        let options = frq_world::F0Options {
            f0_floor_hz: self.config.floor_hz,
            f0_ceiling_hz: self.config.ceiling_hz,
            frame_period_ms,
        };
        let track = frq_world::estimate_f0(estimator, samples, sample_rate, &options)
            .map_err(|error| GeneratorError::Estimation(error.to_string()))?;
        Ok(F0Track {
            frame_period_ms: track.frame_period_ms,
            temporal_positions: track.temporal_positions,
            f0_hz: track.f0_hz,
        })
    }

    fn refine_stonemask(
        &self,
        samples: &[f64],
        sample_rate: u32,
        track: &mut F0Track,
    ) -> Result<(), GeneratorError> {
        let mut world_track = frq_world::F0Track {
            frame_period_ms: track.frame_period_ms,
            temporal_positions: std::mem::take(&mut track.temporal_positions),
            f0_hz: std::mem::take(&mut track.f0_hz),
        };
        frq_world::refine_f0_stonemask(samples, sample_rate, &mut world_track)
            .map_err(|error| GeneratorError::Estimation(error.to_string()))?;
        track.temporal_positions = world_track.temporal_positions;
        track.f0_hz = world_track.f0_hz;
        Ok(())
    }
}

/// Table formats a run can write.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Target {
    Frq,
    Pmk,
    Mrq,
}

/// Everything [`generate`] needs.
#[derive(Debug, Clone)]
pub struct GenerateOptions {
    /// A voicebank folder (recursive scan) or a single `.wav`.
    pub root: PathBuf,
    /// Formats to write; empty is a [`GeneratorError::Config`].
    pub targets: BTreeSet<Target>,
    /// Regenerate tables that already exist (and delete a stale frq alternate).
    pub overwrite: bool,
    pub f0: F0Config,
    /// Worker threads; `0` = all cores.
    pub jobs: usize,
    /// mrq sharing flag (#10): also key entries by the Japanese-side name.
    pub sharing: Option<Sharing>,
    /// Delete `.llsm` for each wav whose mrq f0 entry was written: frq is not
    /// moresampler's f0 source, so frq writes leave the caches alone (#4/#10).
    pub delete_llsm: bool,
}

/// Cooperative cancellation, checked before decode and before writes.
pub type CancelToken = Arc<AtomicBool>;

/// What happened to one wav in a run; the per-file view behind the report
/// lines of #12 and the run rows of #23.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FileReport {
    pub wav: PathBuf,
    /// Targets whose table was written (or, for mrq, upserted and merged).
    pub written: BTreeSet<Target>,
    /// Targets skipped because a table already exists.
    pub existing: BTreeSet<Target>,
    /// Targets the wav cannot produce a table for: a sub-hop wav gets no mrq
    /// entry (#8).
    pub no_entry: BTreeSet<Target>,
    /// The wav had zero frames (#11), so nothing was analyzed or written.
    pub empty: bool,
    /// Cancellation arrived before this wav's writes ran; noted per target.
    pub cancelled: bool,
    /// Non-fatal per-file problems: decode warnings (#11) and pipeline
    /// warnings (failed `.llsm` deletion, failed frq-alt cleanup, ...).
    pub warnings: Vec<String>,
    /// Per-target failure reasons; non-empty means the wav failed.
    pub failures: Vec<String>,
}

impl FileReport {
    pub fn new(wav: PathBuf) -> Self {
        Self {
            wav,
            ..Self::default()
        }
    }

    /// Nothing written, nothing failed and not cancelled.
    pub fn skipped(&self) -> bool {
        self.written.is_empty() && self.failures.is_empty() && !self.cancelled
    }
}

/// Run reporter. Every method is a default no-op so callers implement only
/// what they surface.
///
/// `file_finished` fires once the wav's outcome is final: at the end of its
/// processing for ordinary wavs, and after its folder's `desc.mrq` merge-write
/// for wavs that contributed an mrq entry.
pub trait Progress: Send + Sync {
    fn file_started(&self, _wav: &Path) {}
    fn file_finished(&self, _report: &FileReport) {}
    fn folder_finished(&self, _folder: &Path) {}
    fn finished(&self, _summary: &RunSummary) {}
}

/// Per-run outcome; per-file problems collect here (see [`generate`]).
///
/// A wav with a partial failure (one target failed, another was written) is
/// counted in both `written` and `failed`.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct RunSummary {
    /// Wavs the scan found.
    pub considered: usize,
    /// Wavs with at least one target written.
    pub written: usize,
    /// Wavs with nothing written and no failure (tables existed, or the wav
    /// was zero-length).
    pub skipped: usize,
    /// Per-file failure reasons, in scan order.
    pub failed: Vec<(PathBuf, String)>,
    /// Per-file non-fatal warnings (#11), in scan order.
    pub warnings: Vec<(PathBuf, String)>,
    /// Cancellation stopped the run early; the summary is the partial run.
    pub cancelled: bool,
}

/// The dry-run plan for a run ([`plan`]): the scan's wavs and the tables that
/// already exist per the sidecar rules.
///
/// The plan is existence-only: the caller applies the overwrite policy and
/// reports would-write / would-skip per target.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunPlan {
    pub files: Vec<FilePlan>,
}

/// One wav's plan: the wav and the selected targets that already exist, so the
/// caller can decide would-write / would-skip under its overwrite policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FilePlan {
    pub wav: PathBuf,
    /// Targets whose table already exists per the sidecar rules (frq canonical
    /// or alt, pmk canonical, mrq entry keys incl. the sharing flag).
    pub existing: BTreeSet<Target>,
    /// Non-fatal notes, e.g. a corrupt `desc.mrq` that would be renamed aside.
    pub warnings: Vec<String>,
}

/// Fatal run errors; per-file failures live in [`RunSummary::failed`].
#[derive(Debug)]
pub enum GeneratorError {
    /// The run configuration or root argument is invalid.
    Config(String),
    /// Walking the root failed.
    Scan {
        path: PathBuf,
        error: std::io::Error,
    },
    /// The root holds no wav files.
    NoWavs { root: PathBuf },
    /// f0 estimation failed (transported per file by the pipeline).
    Estimation(String),
}

impl std::fmt::Display for GeneratorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GeneratorError::Config(message) => write!(f, "invalid configuration: {message}"),
            GeneratorError::Scan { path, error } => {
                write!(f, "cannot scan {}: {error}", path.display())
            }
            GeneratorError::NoWavs { root } => {
                write!(f, "no wav files found under {}", root.display())
            }
            GeneratorError::Estimation(message) => write!(f, "f0 estimation failed: {message}"),
        }
    }
}

impl std::error::Error for GeneratorError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            GeneratorError::Scan { error, .. } => Some(error),
            _ => None,
        }
    }
}
