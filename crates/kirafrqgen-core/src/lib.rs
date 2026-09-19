//! The KiraFrqGen generation pipeline: scan a voicebank for wavs, estimate f0
//! through the [`F0Estimator`] seam, and write frequency tables via
//! kirafrq-formats.
//!
//! [`generate`] implements #11's input policy, #8's table policy and the batch
//! semantics of #12; [`generate_wavs`] runs the same pass over an explicit wav
//! list; [`plan`] is the dry-run path (scan plus existence checks, no decode
//! or analysis).
//!
//! Estimators are selected through [`Estimator`] and built by
//! [`build_estimator`] (the feature-gated factory): the WORLD pair always,
//! plus RMVPE when the `ml` feature is on (#48/#49).

pub mod llsm;

mod paths;
mod run;
mod scan;
mod table;
mod voicing;

pub use kirafrq_formats::mrq::Sharing;
pub use kirafrq_world_binding::{FrameObserver, ProgressStage};
pub use run::{generate, generate_wavs, plan};

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

/// f0 estimator selection for [`F0Config`].
///
/// Data-only: the ML variant names no provider type, so the compat build
/// (`--no-default-features`) compiles it too and rejects it at
/// [`build_estimator`] time (#48).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Estimator {
    Dio,
    Harvest,
    Rmvpe,
}

/// ML estimator settings (#48): the model file and the confidence threshold.
/// Non-`Copy` because the path is owned; the pipeline keeps it in
/// [`GenerateOptions`] and hands a clone to the provider.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MlConfig {
    /// The `rmvpe.onnx` file; `None` resolves at factory time (executable
    /// directory, then `KIRAFRQ_ML_DIR`).
    pub model_path: Option<PathBuf>,
    /// The #53 confidence override; `None` uses the model's own default
    /// (RMVPE 0.03). Internal, no CLI/GUI knob.
    pub confidence_threshold: Option<f64>,
}

/// f0 estimation settings shared by every wav in a run.
#[derive(Debug, Clone, PartialEq)]
pub struct F0Config {
    pub estimator: Estimator,
    pub floor_hz: f64,
    pub ceiling_hz: f64,
    pub stone_mask: bool,
    /// The tuned WORLD path (#46/#54): when false the estimator's output is
    /// written as-is, with no energy voicing gate. CLI/GUI surfacing is #49.
    pub world_quirks: bool,
    /// The energy voicing gate's threshold (#54): a voiced frame quieter than
    /// this share of the file's p90 voiced-frame amplitude is forced unvoiced.
    /// Internal, kept out of the headline API.
    #[doc(hidden)]
    pub energy_gate_ratio: f64,
    /// The ML estimator's model and threshold (#48/#53); consulted only when
    /// [`Estimator::Rmvpe`] is selected.
    pub ml: MlConfig,
}

impl Default for F0Config {
    fn default() -> Self {
        // Standing defaults (map Notes): Harvest with StoneMask, 71-800 Hz.
        Self {
            estimator: Estimator::Harvest,
            floor_hz: 71.0,
            ceiling_hz: 800.0,
            stone_mask: true,
            world_quirks: true,
            energy_gate_ratio: 0.05,
            ml: MlConfig::default(),
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
///
/// The optional observer carries frame progress out of the analysis (#34);
/// `None` means no reporting and no added overhead.
pub trait F0Estimator: Send + Sync {
    fn estimate(
        &self,
        samples: &[f64],
        sample_rate: u32,
        frame_period_ms: f64,
        observer: Option<&dyn FrameObserver>,
    ) -> Result<F0Track, GeneratorError>;

    /// StoneMask refinement; estimators without it keep the default no-op.
    fn refine_stonemask(
        &self,
        _samples: &[f64],
        _sample_rate: u32,
        _track: &mut F0Track,
        _observer: Option<&dyn FrameObserver>,
    ) -> Result<(), GeneratorError> {
        Ok(())
    }

    /// Whether this estimator refines with StoneMask (#48). The pipeline
    /// consults this before `refine_stonemask`, so an ML estimator never
    /// refines even when `stone_mask` is set. Defaults to `false`: an
    /// estimator must opt in.
    fn supports_stonemask(&self) -> bool {
        false
    }
}

/// [`F0Estimator`] backed by kirafrq-world-binding (DIO / Harvest / StoneMask).
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
        observer: Option<&dyn FrameObserver>,
    ) -> Result<F0Track, GeneratorError> {
        let estimator = match self.config.estimator {
            Estimator::Dio => kirafrq_world_binding::Estimator::Dio,
            Estimator::Harvest => kirafrq_world_binding::Estimator::Harvest,
            // The factory rejects RMVPE before a WorldEstimator is built.
            Estimator::Rmvpe => {
                return Err(GeneratorError::Config(
                    "RMVPE is not a WORLD estimator".to_string(),
                ));
            }
        };
        let options = kirafrq_world_binding::F0Options {
            f0_floor_hz: self.config.floor_hz,
            f0_ceiling_hz: self.config.ceiling_hz,
            frame_period_ms,
        };
        let track = kirafrq_world_binding::estimate_f0_with_observer(
            estimator,
            samples,
            sample_rate,
            &options,
            observer,
        )
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
        observer: Option<&dyn FrameObserver>,
    ) -> Result<(), GeneratorError> {
        let mut world_track = kirafrq_world_binding::F0Track {
            frame_period_ms: track.frame_period_ms,
            temporal_positions: std::mem::take(&mut track.temporal_positions),
            f0_hz: std::mem::take(&mut track.f0_hz),
        };
        kirafrq_world_binding::refine_f0_stonemask_with_observer(
            samples,
            sample_rate,
            &mut world_track,
            observer,
        )
        .map_err(|error| GeneratorError::Estimation(error.to_string()))?;
        track.temporal_positions = world_track.temporal_positions;
        track.f0_hz = world_track.f0_hz;
        Ok(())
    }

    fn supports_stonemask(&self) -> bool {
        true
    }
}

/// The RMVPE estimator over the ML provider (`ml` builds only).
#[cfg(feature = "ml")]
pub struct MlEstimator {
    inner: kirafrq_ml_provider::rmvpe::Rmvpe,
    model_path: PathBuf,
}

#[cfg(feature = "ml")]
impl MlEstimator {
    /// The model path this estimator will load, for callers that surface it
    /// (the CLI's error hint, the GUI's availability check).
    pub fn model_path(&self) -> &Path {
        &self.model_path
    }
}

#[cfg(feature = "ml")]
impl F0Estimator for MlEstimator {
    fn estimate(
        &self,
        samples: &[f64],
        _sample_rate: u32,
        _frame_period_ms: f64,
        observer: Option<&dyn FrameObserver>,
    ) -> Result<F0Track, GeneratorError> {
        let progress = observer.map(MlProgress);
        let progress = progress
            .as_ref()
            .map(|progress| progress as &dyn kirafrq_ml_provider::ProgressObserver);
        let track = self
            .inner
            .estimate(samples, progress)
            .map_err(|error| GeneratorError::Estimation(error.to_string()))?;
        Ok(F0Track {
            frame_period_ms: track.frame_period_ms,
            temporal_positions: track.temporal_positions,
            f0_hz: track.f0_hz,
        })
    }
}

/// Adapts core's frame observer onto the provider's coarse stage ticks (#48:
/// at most coarse progress, no new machinery).
#[cfg(feature = "ml")]
struct MlProgress<'a>(&'a dyn FrameObserver);

#[cfg(feature = "ml")]
impl kirafrq_ml_provider::ProgressObserver for MlProgress<'_> {
    fn report(&self, done: usize, total: usize) {
        self.0.report(ProgressStage::Estimate, done, total);
    }
}

/// Build the estimator for `config`, resolving the ML model file up front
/// (#48/#49): selecting RMVPE without the `ml` feature or without a model
/// file is an early [`GeneratorError::Config`] naming the expected file, the
/// lookup directories and a download hint.
///
/// `jobs` is the run's single "cores" knob ([`GenerateOptions::jobs`]); the
/// ML session maps it onto ONNX Runtime's intra-op threads.
pub fn build_estimator(
    config: &F0Config,
    jobs: usize,
) -> Result<Box<dyn F0Estimator>, GeneratorError> {
    match config.estimator {
        Estimator::Dio | Estimator::Harvest => Ok(Box::new(WorldEstimator::new(config.clone()))),
        Estimator::Rmvpe => build_ml_estimator(config, jobs),
    }
}

#[cfg(feature = "ml")]
fn build_ml_estimator(
    config: &F0Config,
    jobs: usize,
) -> Result<Box<dyn F0Estimator>, GeneratorError> {
    let model_path = match &config.ml.model_path {
        Some(path) => {
            if !path.is_file() {
                return Err(GeneratorError::Config(format!(
                    "the ML model file {} does not exist",
                    path.display()
                )));
            }
            path.clone()
        }
        None => kirafrq_ml_provider::resolve_model()
            .map_err(|error| GeneratorError::Config(format!("{error}; {ML_DOWNLOAD_HINT}")))?,
    };
    let policy = kirafrq_ml_provider::UvPolicy {
        confidence_threshold: config
            .ml
            .confidence_threshold
            .unwrap_or(kirafrq_ml_provider::RMVPE_DEFAULT_CONFIDENCE_THRESHOLD),
        floor_hz: config.floor_hz,
        ceiling_hz: config.ceiling_hz,
    };
    Ok(Box::new(MlEstimator {
        inner: kirafrq_ml_provider::rmvpe::Rmvpe::new(model_path.clone(), policy, jobs),
        model_path,
    }))
}

#[cfg(not(feature = "ml"))]
fn build_ml_estimator(
    _config: &F0Config,
    _jobs: usize,
) -> Result<Box<dyn F0Estimator>, GeneratorError> {
    Err(GeneratorError::Config(format!(
        "this build has no ML estimator support (compiled without the `ml` feature); \
         {ML_DOWNLOAD_HINT}"
    )))
}

/// The download hint the CLI and GUI share for an unavailable ML model (#48).
pub const ML_DOWNLOAD_HINT: &str = "RMVPE weights are not redistributed with KiraFrqGen; \
     download `rmvpe.onnx` (e.g. from the RVC project's HuggingFace mirror \
     `lj1995/VoiceConversionWebUI`) and place it next to the executable or in \
     the directory named by KIRAFRQ_ML_DIR";

/// Whether this build can offer the ML estimator at all (`ml` on).
pub const ML_SUPPORTED: bool = cfg!(feature = "ml");

/// Whether RMVPE can run right now: the feature is on and a model file
/// resolves. Used by the front ends to pick the capability-aware default
/// (#49) and to grey the option out.
pub fn ml_available(config: &F0Config) -> bool {
    match &config.ml.model_path {
        Some(path) => path.is_file(),
        None => {
            #[cfg(feature = "ml")]
            {
                kirafrq_ml_provider::resolve_model().is_ok()
            }
            #[cfg(not(feature = "ml"))]
            {
                false
            }
        }
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
    /// Whether the reporter wants determinate per-file progress (#34). Only
    /// reporters that surface it opt in, so everyone else skips the hook
    /// entirely and pays no overhead.
    fn wants_file_progress(&self) -> bool {
        false
    }
    /// One wav's analysis fraction, in permille: `done` of `total` (= 1000).
    /// The pipeline composes the estimate and refine phases and latches each,
    /// so the pair never moves backwards; the analysis owns the first 90%
    /// and the write phase the last 10%, one share per selected target
    /// credited as that target's outcome lands (an mrq contribution credits
    /// at its folder merge). Wavs that skip analysis emit no event.
    fn file_progress(&self, _wav: &Path, _done: u64, _total: u64) {}
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
