//! The KiraFrqGen generation pipeline: scan a voicebank for wavs, estimate f0
//! through the [`F0Estimator`] seam, and write frequency tables via
//! kira-frq-core.
//!
//! Skeleton per the #7 resolution. The pipeline never names WORLD;
//! [`WorldEstimator`] is the only place the WORLD wrapper enters. `generate`
//! itself lands with the pipeline ticket (input policy is #11).

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

/// f0 estimator selection for [`F0Config`], in the pipeline's own vocabulary
/// (it never names WORLD).
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

/// [`F0Estimator`] backed by kira-frq-world (DIO / Harvest / StoneMask).
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
            Estimator::Dio => kira_frq_world::Estimator::Dio,
            Estimator::Harvest => kira_frq_world::Estimator::Harvest,
        };
        let options = kira_frq_world::F0Options {
            f0_floor_hz: self.config.floor_hz,
            f0_ceiling_hz: self.config.ceiling_hz,
            frame_period_ms,
        };
        let track = kira_frq_world::estimate_f0(estimator, samples, sample_rate, &options)
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
        let mut world_track = kira_frq_world::F0Track {
            frame_period_ms: track.frame_period_ms,
            temporal_positions: std::mem::take(&mut track.temporal_positions),
            f0_hz: std::mem::take(&mut track.f0_hz),
        };
        kira_frq_world::refine_f0_stonemask(samples, sample_rate, &mut world_track)
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
    pub root: PathBuf,
    pub targets: BTreeSet<Target>,
    pub overwrite: bool,
    pub f0: F0Config,
    /// `0` = all cores.
    pub jobs: usize,
}

/// Cooperative cancellation, checked before decode and before writes.
pub type CancelToken = Arc<AtomicBool>;

/// Run reporter. Every method is a default no-op so callers implement only
/// what they surface.
pub trait Progress: Send + Sync {
    fn file_started(&self, _wav: &Path) {}
    fn file_finished(&self, _wav: &Path) {}
    fn folder_finished(&self, _folder: &Path) {}
    fn finished(&self, _summary: &RunSummary) {}
}

/// Per-run outcome. Per-file failures collect here and never abort the run
/// (#7); only scan/config problems are fatal (`Err`).
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct RunSummary {
    pub considered: usize,
    pub written: usize,
    pub skipped: usize,
    pub failed: Vec<(PathBuf, String)>,
    pub cancelled: bool,
}

/// Fatal run errors; per-file failures live in [`RunSummary::failed`].
#[derive(Debug)]
pub enum GeneratorError {
    /// The run configuration is invalid.
    Config(String),
    /// Scanning the voicebank root failed.
    Scan(std::io::Error),
    /// f0 estimation failed fatally.
    Estimation(String),
}

impl std::fmt::Display for GeneratorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GeneratorError::Config(message) => write!(f, "invalid configuration: {message}"),
            GeneratorError::Scan(error) => write!(f, "failed to scan the voicebank root: {error}"),
            GeneratorError::Estimation(message) => write!(f, "f0 estimation failed: {message}"),
        }
    }
}

impl std::error::Error for GeneratorError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            GeneratorError::Scan(error) => Some(error),
            _ => None,
        }
    }
}

/// Run the generation pass over `opts.root`. Lands with the pipeline ticket
/// once input policy (#11) closes.
pub fn generate(
    opts: &GenerateOptions,
    estimator: &dyn F0Estimator,
    progress: &dyn Progress,
    cancel: &CancelToken,
) -> Result<RunSummary, GeneratorError> {
    let _ = (opts, estimator, progress, cancel);
    todo!("pipeline lands with its build ticket (input policy is #11)")
}
