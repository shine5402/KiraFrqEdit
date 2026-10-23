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
//! plus RMVPE and SwiftF0 when the `ml` feature is on (#48/#49/#69).

pub mod llsm;

mod paths;
#[cfg(feature = "ml")]
mod remote;
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
/// Data-only: the ML variants name no provider type, so the compat build
/// (`--no-default-features`) compiles them too and rejects them at
/// [`build_estimator`] time (#48).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Estimator {
    Dio,
    Harvest,
    Rmvpe,
    SwiftF0,
}

impl Estimator {
    /// Whether recommended tuning's energy voicing gate (#54) applies: every
    /// estimator except RMVPE, whose model confidence is its own voicing
    /// policy (#53/#62/#70).
    pub fn supports_energy_gate(self) -> bool {
        !matches!(self, Estimator::Rmvpe)
    }

    /// The user-facing one-liner the front ends show for this estimator.
    pub fn description(self) -> &'static str {
        match self {
            Estimator::Dio => {
                "Fast traditional DSP-based algorithm from WORLD, but it may struggle on \
                 less-than-ideal recordings."
            }
            Estimator::Harvest => {
                "High quality and noise resistant, but very slow. A traditional DSP-based \
                 algorithm from WORLD."
            }
            Estimator::Rmvpe => {
                "Fast and reliable ML-based estimator, a really good fit for singing material \
                 such as UTAU voicebanks."
            }
            Estimator::SwiftF0 => {
                "Compact and super fast ML-based estimator. Results are less ideal than RMVPE, \
                 but still good."
            }
        }
    }
}

/// The capability-aware default (#49/#62/#71): RMVPE when its model resolves,
/// else SwiftF0 in an ML build, else Harvest. The front ends resolve this once
/// at startup; an explicit choice always wins.
///
/// SwiftF0 ships bundled, so a modern build is effectively RMVPE-else-SwiftF0;
/// the compat build (`--no-default-features`) falls back Harvest-else-DIO.
/// `rmvpe_available` is [`ml_available`]'s result (RMVPE's model resolves).
pub fn default_estimator(rmvpe_available: bool) -> Estimator {
    if rmvpe_available {
        Estimator::Rmvpe
    } else if ML_SUPPORTED {
        Estimator::SwiftF0
    } else {
        Estimator::Harvest
    }
}

/// Which runtime the ML estimators (RMVPE, SwiftF0) run on.
///
/// Data-only, like [`Estimator`]: every build compiles every variant, and a
/// runner the platform lacks is rejected at [`build_estimator`] time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MlRunner {
    /// ONNX Runtime's CPU path; the fallback wherever no GPU runner can run.
    #[default]
    Cpu,
    /// DirectML (Windows): a DirectX 12 GPU.
    DirectMl,
    /// Core ML (Apple-silicon Mac): the Mac's GPU (#84).
    CoreMl,
    /// An ML server on the network (#85), on whatever runner the server
    /// uses; [`MlConfig::server`] names it.
    Remote,
}

impl MlRunner {
    /// The user-facing one-liner the front ends show for this runner.
    pub fn description(self) -> &'static str {
        match self {
            MlRunner::Cpu => "Runs the ML estimators on the processor.",
            MlRunner::DirectMl => "Runs the ML estimators on a DirectX 12 GPU.",
            MlRunner::CoreMl => "Runs the ML estimators on the Mac's GPU through Core ML.",
            MlRunner::Remote => {
                "Sends ML estimation to an ML server on the network, which runs it on its own \
                 ML runner."
            }
        }
    }

    /// Whether the DirectML device setting applies to this runner.
    pub fn is_directml(self) -> bool {
        matches!(self, MlRunner::DirectMl)
    }

    /// Whether the ML server settings apply to this runner.
    pub fn is_remote(self) -> bool {
        matches!(self, MlRunner::Remote)
    }
}

/// The ML server's default port (#85).
pub const ML_SERVER_DEFAULT_PORT: u16 = 39470;

#[cfg(feature = "ml")]
const _: () = assert!(ML_SERVER_DEFAULT_PORT == kirafrq_ml_remote::DEFAULT_PORT);

/// Where an ML server listens: a host (an IPv6 address without its
/// brackets) and a port.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MlServer {
    pub host: String,
    pub port: u16,
}

impl MlServer {
    /// Parse `HOST[:PORT]`; the port defaults to [`ML_SERVER_DEFAULT_PORT`].
    /// An IPv6 host goes in brackets when a port follows (`[fe80::1]:39470`);
    /// a bare IPv6 address takes the default port.
    pub fn parse(value: &str) -> Result<Self, String> {
        let value = value.trim();
        let port = |text: &str| {
            text.parse::<u16>()
                .ok()
                .filter(|&port| port != 0)
                .ok_or_else(|| format!("`{text}` is not a port (1-65535)"))
        };
        let (host, port) = if let Some(rest) = value.strip_prefix('[') {
            let (host, after) = rest
                .split_once(']')
                .ok_or_else(|| format!("`{value}` has no closing `]`"))?;
            if after.is_empty() {
                (host, ML_SERVER_DEFAULT_PORT)
            } else {
                let text = after
                    .strip_prefix(':')
                    .ok_or_else(|| format!("`{value}` has text after `]` that is not a port"))?;
                (host, port(text)?)
            }
        } else {
            match value.split_once(':') {
                // More than one colon: a bare IPv6 address.
                Some((_, rest)) if rest.contains(':') => (value, ML_SERVER_DEFAULT_PORT),
                Some((host, text)) => (host, port(text)?),
                None => (value, ML_SERVER_DEFAULT_PORT),
            }
        };
        if host.is_empty() || host.chars().any(char::is_whitespace) {
            return Err(format!("`{value}` has no host"));
        }
        Ok(Self {
            host: host.to_string(),
            port,
        })
    }
}

impl std::fmt::Display for MlServer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.host.contains(':') {
            write!(f, "[{}]:{}", self.host, self.port)
        } else {
            write!(f, "{}:{}", self.host, self.port)
        }
    }
}

/// ML estimator settings (#48): the model file, the confidence threshold and
/// the runner. Non-`Copy` because the path is owned; the pipeline keeps
/// it in [`GenerateOptions`] and hands a clone to the provider.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MlConfig {
    /// An explicit model file override; `None` resolves by filename
    /// (`rmvpe.onnx`, `swiftf0.onnx`) via the executable directory, then
    /// `KIRAFRQ_ML_DIR`, then SwiftF0's bundled fallback.
    pub model_path: Option<PathBuf>,
    /// The #53 confidence override; `None` uses the model's own default
    /// (RMVPE 0.03, SwiftF0 0.9). Internal, no CLI/GUI knob.
    pub confidence_threshold: Option<f64>,
    /// Which runtime the ML estimators use.
    pub runner: MlRunner,
    /// The DirectML adapter index (DXGI order; ONNX Runtime calls it
    /// `device_id`). `None` lets ONNX Runtime prefer the high-performance
    /// GPU. Ignored unless `runner` is a DirectML variant.
    pub directml_device: Option<u32>,
    /// The ML server for [`MlRunner::Remote`]; ignored by the other runners.
    pub server: Option<MlServer>,
    /// The ML server's shared token, when it was started with one.
    pub server_token: Option<String>,
}

/// Whether this build can offer the DirectML runner at all (Windows `ml`
/// builds).
#[cfg(feature = "ml")]
pub const DIRECTML_SUPPORTED: bool = kirafrq_ml_provider::DIRECTML_SUPPORTED;
#[cfg(not(feature = "ml"))]
pub const DIRECTML_SUPPORTED: bool = false;

/// Whether this build can offer the Core ML runner at all (Apple-silicon
/// macOS `ml` builds).
#[cfg(feature = "ml")]
pub const COREML_SUPPORTED: bool = kirafrq_ml_provider::COREML_SUPPORTED;
#[cfg(not(feature = "ml"))]
pub const COREML_SUPPORTED: bool = false;

/// One DirectML adapter the runner can target.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MlDevice {
    /// The DXGI adapter index: DirectML's `device_id`.
    pub index: u32,
    /// The adapter description, e.g. `NVIDIA GeForce RTX 3070`.
    pub name: String,
}

impl MlDevice {
    /// The device picker's label, e.g. `0 — NVIDIA GeForce RTX 3070 (default)`.
    pub fn label(&self) -> String {
        if self.index == 0 {
            format!("{} — {} (default)", self.index, self.name)
        } else {
            format!("{} — {}", self.index, self.name)
        }
    }
}

/// The DirectML adapters this machine offers, in `device_id` order. `Err`
/// carries the user-facing reason when the list cannot be produced (no
/// DirectML in this build, or a DXGI failure).
pub fn ml_devices() -> Result<Vec<MlDevice>, String> {
    #[cfg(feature = "ml")]
    {
        kirafrq_ml_provider::directml_devices()
            .map(|devices| {
                devices
                    .into_iter()
                    .map(|device| MlDevice {
                        index: device.index,
                        name: device.name,
                    })
                    .collect()
            })
            .map_err(|error| error.to_string())
    }
    #[cfg(not(feature = "ml"))]
    {
        Err("this build has no ML support".to_string())
    }
}

/// Whether the DirectML runner can run here: a Windows ML build with at least
/// one hardware adapter. DirectML keeps ONNX Runtime's CPU path alongside it,
/// so a host that fails this check is served by the CPU runner instead.
pub fn directml_usable() -> bool {
    DIRECTML_SUPPORTED && ml_devices().is_ok_and(|devices| !devices.is_empty())
}

/// Whether the Core ML runner can run here: an Apple-silicon macOS ML build
/// where a Core ML session commits. The probe builds a small session once per
/// process.
pub fn coreml_usable() -> bool {
    #[cfg(feature = "ml")]
    {
        kirafrq_ml_provider::coreml_usable()
    }
    #[cfg(not(feature = "ml"))]
    {
        false
    }
}

/// Whether `runner` can run here: CPU in every ML build, the GPU runners per
/// [`directml_usable`] and [`coreml_usable`]. Remote is offered by every ML
/// build; whether its server answers is [`ml_server_info`]'s question.
pub fn ml_runner_usable(runner: MlRunner) -> bool {
    match runner {
        MlRunner::Cpu | MlRunner::Remote => ML_SUPPORTED,
        MlRunner::DirectMl => directml_usable(),
        MlRunner::CoreMl => coreml_usable(),
    }
}

/// The capability-aware ML runner default: the platform's GPU runner
/// (DirectML on Windows, Core ML on Apple silicon) when `usable`, else CPU.
/// The front ends resolve this once at startup, passing
/// [`ml_runner_usable`]; an explicit choice always wins.
pub fn default_ml_runner(usable: impl Fn(MlRunner) -> bool) -> MlRunner {
    [MlRunner::DirectMl, MlRunner::CoreMl]
        .into_iter()
        .find(|&runner| usable(runner))
        .unwrap_or(MlRunner::Cpu)
}

/// f0 estimation settings shared by every wav in a run.
#[derive(Debug, Clone, PartialEq)]
pub struct F0Config {
    pub estimator: Estimator,
    pub floor_hz: f64,
    pub ceiling_hz: f64,
    pub stone_mask: bool,
    /// "Apply recommended tuning" (#46/#54/#62): when false the estimator's
    /// output is written as-is, with no energy voicing gate (every estimator
    /// but RMVPE) and no aperiodicity gate (Harvest). CLI/GUI surfacing is
    /// #49/#71.
    pub recommended_tuning: bool,
    /// The energy voicing gate's threshold (#54): a voiced frame quieter than
    /// this share of the file's p90 voiced-frame amplitude is forced unvoiced.
    /// Internal, kept out of the headline API.
    #[doc(hidden)]
    pub energy_gate_ratio: f64,
    /// The aperiodicity gate's threshold (#64): a voiced Harvest frame whose
    /// raw D4C LoveTrain statistic is below this is forced unvoiced. Internal,
    /// kept out of the headline API.
    #[doc(hidden)]
    pub aperiodicity_gate_threshold: f64,
    /// The ML estimators' model and threshold (#48/#53); consulted only when
    /// [`Estimator::Rmvpe`] or [`Estimator::SwiftF0`] is selected.
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
            recommended_tuning: true,
            energy_gate_ratio: 0.05,
            aperiodicity_gate_threshold: 0.85,
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

    /// The raw D4C LoveTrain aperiodicity statistic per frame, for the tuned
    /// WORLD path's aperiodicity gate (#64). `Ok(None)` means the estimator
    /// has no such statistic (the ML tier, or a WORLD estimator that does not
    /// compute it), so the gate is skipped. The pipeline only asks on the
    /// Harvest path; a failure is a per-file failure like StoneMask.
    fn aperiodicity0(
        &self,
        _samples: &[f64],
        _sample_rate: u32,
        _track: &F0Track,
    ) -> Result<Option<Vec<f64>>, GeneratorError> {
        Ok(None)
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
            // The factory rejects the ML estimators before a WorldEstimator
            // is built.
            Estimator::Rmvpe | Estimator::SwiftF0 => {
                return Err(GeneratorError::Config(format!(
                    "the {:?} ML estimator is not a WORLD estimator",
                    self.config.estimator
                )));
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

    fn aperiodicity0(
        &self,
        samples: &[f64],
        sample_rate: u32,
        track: &F0Track,
    ) -> Result<Option<Vec<f64>>, GeneratorError> {
        let statistic = kirafrq_world_binding::d4c_aperiodicity0(
            samples,
            sample_rate,
            &track.temporal_positions,
            &track.f0_hz,
        )
        .map_err(|error| GeneratorError::Estimation(error.to_string()))?;
        Ok(Some(statistic))
    }

    fn supports_stonemask(&self) -> bool {
        true
    }
}

/// The ML estimator over the provider (`ml` builds only): RMVPE or SwiftF0.
#[cfg(feature = "ml")]
pub struct MlEstimator {
    inner: MlModel,
}

#[cfg(feature = "ml")]
enum MlModel {
    Rmvpe(kirafrq_ml_provider::rmvpe::Rmvpe),
    SwiftF0(kirafrq_ml_provider::swiftf0::SwiftF0),
}

#[cfg(feature = "ml")]
impl MlEstimator {
    fn rmvpe(inner: kirafrq_ml_provider::rmvpe::Rmvpe) -> Self {
        Self {
            inner: MlModel::Rmvpe(inner),
        }
    }

    fn swiftf0(inner: kirafrq_ml_provider::swiftf0::SwiftF0) -> Self {
        Self {
            inner: MlModel::SwiftF0(inner),
        }
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
        let track = match &self.inner {
            MlModel::Rmvpe(inner) => inner.estimate(samples, progress),
            MlModel::SwiftF0(inner) => inner.estimate(samples, progress),
        }
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

/// Build the estimator for `config`, resolving the ML model up front
/// (#48/#49/#69): selecting an ML estimator without the `ml` feature, without
/// a resolvable RMVPE file, or with a missing explicit model is an early
/// [`GeneratorError::Config`], not a per-file failure.
///
/// `jobs` is the run's single "cores" knob ([`GenerateOptions::jobs`]); the
/// ML session maps it onto ONNX Runtime's intra-op threads.
pub fn build_estimator(
    config: &F0Config,
    jobs: usize,
) -> Result<Box<dyn F0Estimator>, GeneratorError> {
    match config.estimator {
        Estimator::Dio | Estimator::Harvest => Ok(Box::new(WorldEstimator::new(config.clone()))),
        // The DSP estimators always run here; only ML estimation goes remote.
        #[cfg(feature = "ml")]
        Estimator::Rmvpe | Estimator::SwiftF0 if config.ml.runner.is_remote() => {
            remote::build(config)
        }
        Estimator::Rmvpe => build_ml_estimator(config, jobs),
        Estimator::SwiftF0 => build_swiftf0_estimator(config, jobs),
    }
}

/// An explicit `model_path` override that exists, else `None` so the caller
/// resolves by filename. A set-but-missing path is an early config error
/// naming the estimator.
#[cfg(feature = "ml")]
fn explicit_model_path(
    config: &F0Config,
    estimator: &str,
) -> Result<Option<PathBuf>, GeneratorError> {
    match &config.ml.model_path {
        Some(path) if !path.is_file() => Err(GeneratorError::Config(format!(
            "the {estimator} model file {} does not exist",
            path.display()
        ))),
        Some(path) => Ok(Some(path.clone())),
        None => Ok(None),
    }
}

/// The #53 policy for an ML model: the config's range and an optional
/// threshold override, else the model's own default.
#[cfg(feature = "ml")]
fn ml_policy(config: &F0Config, default_threshold: f64) -> kirafrq_ml_provider::UvPolicy {
    kirafrq_ml_provider::UvPolicy {
        confidence_threshold: config.ml.confidence_threshold.unwrap_or(default_threshold),
        floor_hz: config.floor_hz,
        ceiling_hz: config.ceiling_hz,
    }
}

/// The provider's session settings for the config: the runner and, for
/// DirectML, the device.
#[cfg(feature = "ml")]
fn ml_session(config: &F0Config) -> kirafrq_ml_provider::SessionConfig {
    kirafrq_ml_provider::SessionConfig {
        runner: match config.ml.runner {
            MlRunner::Cpu => kirafrq_ml_provider::Runner::Cpu,
            MlRunner::DirectMl => kirafrq_ml_provider::Runner::DirectMl,
            MlRunner::CoreMl => kirafrq_ml_provider::Runner::CoreMl,
            // The factory sends Remote to the ML server before a local
            // session is ever configured.
            MlRunner::Remote => kirafrq_ml_provider::Runner::Cpu,
        },
        directml_device: config.ml.directml_device,
    }
}

#[cfg(feature = "ml")]
fn build_ml_estimator(
    config: &F0Config,
    jobs: usize,
) -> Result<Box<dyn F0Estimator>, GeneratorError> {
    let model_path = match explicit_model_path(config, "ML")? {
        Some(path) => path,
        None => kirafrq_ml_provider::resolve_model()
            .map_err(|error| GeneratorError::Config(format!("{error}; {ML_DOWNLOAD_HINT}")))?,
    };
    let policy = ml_policy(
        config,
        kirafrq_ml_provider::RMVPE_DEFAULT_CONFIDENCE_THRESHOLD,
    );
    let inner = kirafrq_ml_provider::rmvpe::Rmvpe::new(model_path, policy, jobs)
        .with_session_config(ml_session(config));
    // Load the session now so an unloadable model is the same early config
    // error as a missing one (#49), not a per-file failure.
    inner
        .ensure_session()
        .map_err(|error| GeneratorError::Config(format!("{error}; {ML_DOWNLOAD_HINT}")))?;
    Ok(Box::new(MlEstimator::rmvpe(inner)))
}

/// Build SwiftF0 (#69): an explicit `model_path` overrides the filename
/// lookup, which falls back to the bundled MIT model, so a modern build never
/// needs a download.
#[cfg(feature = "ml")]
fn build_swiftf0_estimator(
    config: &F0Config,
    jobs: usize,
) -> Result<Box<dyn F0Estimator>, GeneratorError> {
    let source = match explicit_model_path(config, "SwiftF0")? {
        Some(path) => kirafrq_ml_provider::ModelSource::File(path),
        None => kirafrq_ml_provider::swiftf0::resolve_source(),
    };
    let policy = ml_policy(
        config,
        kirafrq_ml_provider::SWIFTF0_DEFAULT_CONFIDENCE_THRESHOLD,
    );
    let inner = kirafrq_ml_provider::swiftf0::SwiftF0::new(source, policy, jobs)
        .with_session_config(ml_session(config));
    inner
        .ensure_session()
        .map_err(|error| GeneratorError::Config(error.to_string()))?;
    Ok(Box::new(MlEstimator::swiftf0(inner)))
}

#[cfg(not(feature = "ml"))]
fn build_ml_estimator(
    _config: &F0Config,
    _jobs: usize,
) -> Result<Box<dyn F0Estimator>, GeneratorError> {
    Err(no_ml_support())
}

#[cfg(not(feature = "ml"))]
fn build_swiftf0_estimator(
    _config: &F0Config,
    _jobs: usize,
) -> Result<Box<dyn F0Estimator>, GeneratorError> {
    Err(no_ml_support())
}

#[cfg(not(feature = "ml"))]
fn no_ml_support() -> GeneratorError {
    GeneratorError::Config(
        "this build has no ML estimator support (compiled without the `ml` feature)".to_string(),
    )
}

/// The download hint the CLI and GUI share for an unavailable ML model (#48).
pub const ML_DOWNLOAD_HINT: &str = "RMVPE weights are not redistributed with KiraFrqGen; \
     download `rmvpe.onnx` (e.g. from the RVC project's HuggingFace mirror \
     `lj1995/VoiceConversionWebUI`) and place it next to the executable or in \
     the directory named by KIRAFRQ_ML_DIR";

/// Whether this build can offer the ML estimator at all (`ml` on).
pub const ML_SUPPORTED: bool = cfg!(feature = "ml");

/// Whether RMVPE can run right now: the feature is on and a model file
/// resolves. The front ends use it for the capability-aware default (#49) and
/// for the model-missing UX (#72). This is the local answer; with
/// [`MlRunner::Remote`] the front ends ask [`ml_server_info`] instead.
pub fn ml_available(config: &F0Config) -> bool {
    if !ML_SUPPORTED {
        return false;
    }
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

/// What an ML server reported (#85), for the front ends' availability,
/// default estimator and connection test.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MlServerInfo {
    /// One line: the server's KiraFrqGen version, ML runner and device, and
    /// its ML estimators.
    pub summary: String,
    /// The server has RMVPE's model file.
    pub rmvpe: bool,
    /// The server can run SwiftF0.
    pub swiftf0: bool,
    /// The server runs another KiraFrqGen version under the same protocol;
    /// a warning, not an error.
    pub version_warning: Option<String>,
}

impl MlServerInfo {
    /// Whether the server can run `estimator`; the DSP estimators never run
    /// there, so they are always `false`.
    pub fn available(&self, estimator: Estimator) -> bool {
        match estimator {
            Estimator::Rmvpe => self.rmvpe,
            Estimator::SwiftF0 => self.swiftf0,
            Estimator::Dio | Estimator::Harvest => false,
        }
    }
}

/// Ask `config`'s ML server what it is and whether it accepts the token. An
/// unreachable server, a protocol mismatch and a missing or wrong token are
/// `Err` with the user-facing reason. Blocks for up to a few seconds.
pub fn ml_server_info(config: &MlConfig) -> Result<MlServerInfo, String> {
    #[cfg(feature = "ml")]
    {
        remote::server_info(config)
    }
    #[cfg(not(feature = "ml"))]
    {
        let _ = config;
        Err("this build has no ML support".to_string())
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
    /// Targets whose table was written (for mrq, once the folder's batch write
    /// that carries the entry has landed).
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
/// processing, except for a wav with an mrq entry, which waits for the
/// folder's `desc.mrq` write that persists the entry (a batch flush or the
/// run's tail write).
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
    /// when the write that persists its entry lands). Wavs that skip analysis
    /// emit no event.
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

/// The dry-run plan for a run ([`plan`]): the scan's wavs, the tables that
/// already exist per the sidecar rules, and which of those are stale against
/// the wav's mtime.
///
/// The plan is existence-and-staleness only: the caller applies the overwrite
/// policy and reports would-write / would-skip per target.
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
    /// Existing targets whose table is older than the wav, so a fill-missing
    /// run would keep a stale table. frq and pmk compare file mtimes (frq
    /// counts fresh when either spelling is); mrq follows moresampler's own
    /// rule, `timestamp != 0 && wav_mtime > timestamp` in whole seconds. Always
    /// a subset of [`existing`](Self::existing).
    pub outdated: BTreeSet<Target>,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_estimator_prefers_rmvpe_then_swiftf0_then_harvest() {
        // #62/#71: RMVPE when its model resolves, else SwiftF0 in an ML build,
        // else Harvest. DIO is selectable but never the default.
        assert_eq!(default_estimator(true), Estimator::Rmvpe);
        let fallback = if ML_SUPPORTED {
            Estimator::SwiftF0
        } else {
            Estimator::Harvest
        };
        assert_eq!(default_estimator(false), fallback);
        assert_ne!(fallback, Estimator::Dio);
    }

    #[test]
    fn the_rmvpe_description_does_not_mention_a_required_model() {
        let description = Estimator::Rmvpe.description();
        assert!(!description.contains("Requires model"));
        assert!(description.contains("singing material"));
    }

    #[test]
    fn every_estimator_has_a_description() {
        for estimator in [
            Estimator::Dio,
            Estimator::Harvest,
            Estimator::Rmvpe,
            Estimator::SwiftF0,
        ] {
            assert!(!estimator.description().is_empty());
        }
    }

    #[test]
    fn the_descriptions_match_the_62_wording() {
        assert_eq!(
            Estimator::Rmvpe.description(),
            "Fast and reliable ML-based estimator, a really good fit for singing material \
             such as UTAU voicebanks."
        );
        assert_eq!(
            Estimator::SwiftF0.description(),
            "Compact and super fast ML-based estimator. Results are less ideal than RMVPE, \
             but still good."
        );
        assert_eq!(
            Estimator::Harvest.description(),
            "High quality and noise resistant, but very slow. A traditional DSP-based \
             algorithm from WORLD."
        );
        assert_eq!(
            Estimator::Dio.description(),
            "Fast traditional DSP-based algorithm from WORLD, but it may struggle on \
             less-than-ideal recordings."
        );
    }

    #[test]
    fn the_default_ml_config_runs_on_the_cpu_with_an_automatic_device() {
        let ml = MlConfig::default();
        assert_eq!(ml.runner, MlRunner::Cpu);
        assert_eq!(ml.directml_device, None, "None means ONNX Runtime picks");
    }

    #[test]
    fn every_ml_runner_has_a_description() {
        for runner in [
            MlRunner::Cpu,
            MlRunner::DirectMl,
            MlRunner::CoreMl,
            MlRunner::Remote,
        ] {
            assert!(!runner.description().is_empty());
        }
    }

    #[test]
    fn the_ml_runner_default_prefers_the_usable_gpu_runner() {
        assert_eq!(
            default_ml_runner(|runner| runner == MlRunner::DirectMl),
            MlRunner::DirectMl
        );
        assert_eq!(
            default_ml_runner(|runner| runner == MlRunner::CoreMl),
            MlRunner::CoreMl
        );
        assert_eq!(default_ml_runner(|_| false), MlRunner::Cpu);
    }

    #[test]
    fn only_the_platform_gpu_runner_can_be_usable() {
        assert!(ml_runner_usable(MlRunner::Cpu) == ML_SUPPORTED);
        if !DIRECTML_SUPPORTED {
            assert!(!ml_runner_usable(MlRunner::DirectMl));
        }
        if !COREML_SUPPORTED {
            assert!(!ml_runner_usable(MlRunner::CoreMl));
        }
    }

    #[test]
    fn device_labels_mark_the_default_adapter() {
        let first = MlDevice {
            index: 0,
            name: "Integrated Graphics".to_owned(),
        };
        assert_eq!(first.label(), "0 — Integrated Graphics (default)");

        let second = MlDevice {
            index: 2,
            name: "Discrete GPU".to_owned(),
        };
        assert_eq!(second.label(), "2 — Discrete GPU");
    }

    #[test]
    fn only_directml_uses_a_device() {
        assert!(!MlRunner::Cpu.is_directml());
        assert!(MlRunner::DirectMl.is_directml());
        assert!(!MlRunner::CoreMl.is_directml());
    }

    #[cfg(feature = "ml")]
    #[test]
    fn the_ml_runner_maps_onto_the_provider_session() {
        let config = F0Config::default();
        assert_eq!(
            ml_session(&config),
            kirafrq_ml_provider::SessionConfig::default()
        );

        let config = F0Config {
            ml: MlConfig {
                runner: MlRunner::DirectMl,
                directml_device: Some(2),
                ..MlConfig::default()
            },
            ..F0Config::default()
        };
        assert_eq!(
            ml_session(&config),
            kirafrq_ml_provider::SessionConfig {
                runner: kirafrq_ml_provider::Runner::DirectMl,
                directml_device: Some(2),
            }
        );
        let config = F0Config {
            ml: MlConfig {
                runner: MlRunner::CoreMl,
                ..MlConfig::default()
            },
            ..F0Config::default()
        };
        assert_eq!(
            ml_session(&config).runner,
            kirafrq_ml_provider::Runner::CoreMl
        );
    }

    #[test]
    fn the_world_estimator_exposes_the_d4c_statistic_per_frame() {
        let sample_rate = 44_100u32;
        let samples: Vec<f64> = (0..sample_rate)
            .map(|n| (2.0 * std::f64::consts::PI * 220.0 * n as f64 / sample_rate as f64).sin())
            .collect();
        let estimator = WorldEstimator::new(F0Config::default());
        let track = estimator
            .estimate(&samples, sample_rate, crate::table::frame_period_ms(), None)
            .unwrap();

        let statistic = estimator
            .aperiodicity0(&samples, sample_rate, &track)
            .unwrap()
            .expect("the WORLD estimator provides the statistic");

        assert_eq!(statistic.len(), track.f0_hz.len());
    }
}
