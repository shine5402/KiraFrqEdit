//! SwiftF0 over ONNX Runtime (#69): a raw-audio adapter, the #53 policy
//! wiring and the lazily-built session.
//!
//! Contract (verified in #59 against `swift_f0/core.py` and the committed
//! model):
//!
//! - input: raw mono `[1, L]` f32 at 16 kHz; the graph owns the STFT
//!   frontend, so there is no mel preprocessing. Inputs shorter than
//!   [`MIN_AUDIO_LENGTH`] are zero-padded: the STFT kernel rejects them.
//! - output: `pitch_hz` and `confidence`, one value per native frame.
//! - native frame `k` sits at `(127.5 + k*256) / 16000` s
//!   ([`crate::grid::SWIFTF0_CONTRACT`]).
//!
//! The `native_*` methods expose the model's own grid for local verification
//! against the Python `swift_f0` reference; they are not a pipeline surface.

use std::sync::Arc;

use ort::session::Session;
use ort::value::Tensor;

use crate::grid::{NativeContour, SWIFTF0_CONTRACT, TABLE_HOP_SAMPLES, Track, map_to_table_grid};
use crate::model::{ModelSource, SWIFTF0_FILE_NAME, resolve_file};
use crate::policy::UvPolicy;
use crate::resample::resample_to_model_rate;
use crate::runner::SessionConfig;
use crate::session;
use crate::{Error, ProgressObserver};

/// The reference wrapper's minimum input length (`SwiftF0.MIN_AUDIO_LENGTH`):
/// a shorter 16 kHz input fails inside ONNX Runtime's STFT kernel, so it is
/// zero-padded up to this (#59).
pub const MIN_AUDIO_LENGTH: usize = 256;

/// The bundled MIT model, pinned to the upstream commit
/// `64700fce8ef39c2970814bf427ac1d75a2f20d72` (#59). An on-disk
/// `swiftf0.onnx` overrides it.
pub const BUNDLED_MODEL: &[u8] = include_bytes!("../assets/swiftf0.onnx");

/// Progress ticks per estimate: resample, inference, map.
const PROGRESS_TICKS: usize = 4;

/// The output tensor names the graph emits.
const PITCH_OUTPUT: &str = "pitch_hz";
const CONFIDENCE_OUTPUT: &str = "confidence";

/// Validate that an output tensor's shape is `[1, frames]`, returning
/// `frames`.
fn validate_output_shape(name: &str, shape: &[i64]) -> Result<usize, Error> {
    match shape {
        [1, frames] => Ok(*frames as usize),
        _ => Err(Error::Ort(format!(
            "SwiftF0 `{name}` shape {shape:?} is not [1, frames]"
        ))),
    }
}

/// Resolve SwiftF0's model: the on-disk `swiftf0.onnx` (executable directory,
/// then `KIRAFRQ_ML_DIR`) if present, else the bundled bytes.
pub fn resolve_source() -> ModelSource {
    match resolve_file(SWIFTF0_FILE_NAME) {
        Ok(path) => ModelSource::File(path),
        Err(_) => ModelSource::Bundled {
            file_name: SWIFTF0_FILE_NAME,
            bytes: BUNDLED_MODEL,
        },
    }
}

/// The SwiftF0 estimator. By default it runs on the process-wide
/// session, built lazily (see [`crate::session`]);
/// [`SwiftF0::with_own_session`] gives it its own.
#[derive(Clone)]
pub struct SwiftF0 {
    source: ModelSource,
    policy: UvPolicy,
    /// `intra_op_num_threads`; `0` leaves ONNX Runtime's default (physical
    /// cores). Clamped against the machine when the session is built.
    jobs: usize,
    /// The execution provider and device; CPU by default.
    session_config: SessionConfig,
    /// An own session ([`Self::with_own_session`]); `None` uses the
    /// process-wide one.
    session: Option<Arc<session::Owned>>,
}

impl SwiftF0 {
    /// The estimator for `source` with the #53 policy; `jobs` follows the
    /// run's single "cores" knob (0 = ONNX Runtime default).
    pub fn new(source: impl Into<ModelSource>, policy: UvPolicy, jobs: usize) -> Self {
        Self {
            source: source.into(),
            policy,
            jobs,
            session_config: SessionConfig::default(),
            session: None,
        }
    }

    /// Run this estimator with `session_config`'s execution provider and
    /// device (CPU by default).
    #[must_use]
    pub fn with_session_config(mut self, session_config: SessionConfig) -> Self {
        self.session_config = session_config;
        self
    }

    /// Build the process session now if it is not built yet, so an unloadable
    /// model fails at configuration time (#49) instead of on the first file.
    pub fn ensure_session(&self) -> Result<(), Error> {
        self.run(|_| Ok(()))
    }

    /// Give this estimator its own session, built now, instead of the
    /// process-wide one: the ML server keeps both ML estimators loaded. Clones
    /// share it. Call it after [`Self::with_session_config`]: the session is
    /// built for the settings at hand.
    pub fn with_own_session(mut self) -> Result<Self, Error> {
        self.session = Some(Arc::new(session::Owned::build(
            &self.source,
            self.jobs,
            self.session_config,
        )?));
        Ok(self)
    }

    /// The same estimator, session included, with another #53 policy: the ML
    /// server takes the range and threshold from each request.
    #[must_use]
    pub fn with_policy(mut self, policy: UvPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Run `f` on this estimator's session: its own, else the process-wide
    /// one.
    fn run<R>(&self, f: impl FnOnce(&mut Session) -> Result<R, Error>) -> Result<R, Error> {
        match &self.session {
            Some(session) => session.run(f),
            None => session::with(&self.source, self.jobs, self.session_config, f),
        }
    }

    /// Estimate f0 for mono 44.1 kHz `samples` and map it onto the table grid
    /// (#47): resample to 16 kHz, run SwiftF0, apply the #53 policy on the
    /// native grid, then interpolate onto the 5.805 ms table frames.
    ///
    /// `samples.len() < 256` short-circuits to all-unvoiced without touching
    /// the model: the only table frame is the forced-unvoiced trailing one.
    pub fn estimate(
        &self,
        samples: &[f64],
        observer: Option<&dyn ProgressObserver>,
    ) -> Result<Track, Error> {
        let report = |done: usize| {
            if let Some(observer) = observer {
                observer.report(done, PROGRESS_TICKS);
            }
        };
        report(0);
        if samples.len() < TABLE_HOP_SAMPLES {
            return Ok(map_to_table_grid(
                &NativeContour {
                    contract: SWIFTF0_CONTRACT,
                    f0_hz: Vec::new(),
                },
                samples.len(),
            ));
        }

        let audio = resample_to_model_rate(samples, crate::grid::TABLE_SAMPLE_RATE, 16_000)?;
        let audio: Vec<f32> = audio.into_iter().map(|sample| sample as f32).collect();
        report(1);

        let native_f0 = self.infer(&audio)?;
        report(3);

        let contour = NativeContour {
            contract: SWIFTF0_CONTRACT,
            f0_hz: native_f0,
        };
        let track = map_to_table_grid(&contour, samples.len());
        report(PROGRESS_TICKS);
        Ok(track)
    }

    /// The policy-applied f0 on the model's native grid, for local
    /// verification probes.
    #[doc(hidden)]
    pub fn native_estimate(&self, audio_16k: &[f32]) -> Result<Vec<f64>, Error> {
        self.infer(audio_16k)
    }

    /// The raw `pitch_hz` and `confidence` the graph emits on the native grid,
    /// with the mandatory zero-pad applied, before the #53 policy.
    #[doc(hidden)]
    pub fn native_outputs(&self, audio_16k: &[f32]) -> Result<(Vec<f32>, Vec<f32>), Error> {
        let mut audio = audio_16k.to_vec();
        if audio.len() < MIN_AUDIO_LENGTH {
            audio.resize(MIN_AUDIO_LENGTH, 0.0);
        }

        self.run(|session| outputs_on(session, &audio))
    }

    /// Run one inference now, so the first real estimate pays no warmup (the
    /// ML server's startup).
    pub fn warm(&self) -> Result<(), Error> {
        let audio = vec![0.0; 16_000];
        self.run(|session| outputs_on(session, &audio).map(|_| ()))
    }

    /// One full inference: the raw outputs with the #53 policy applied on the
    /// native grid.
    fn infer(&self, audio_16k: &[f32]) -> Result<Vec<f64>, Error> {
        let (pitch, confidence) = self.native_outputs(audio_16k)?;
        Ok(pitch
            .iter()
            .zip(&confidence)
            .map(|(&f0, &confidence)| self.policy.apply(f64::from(f0), f64::from(confidence)))
            .collect())
    }
}

/// [`SwiftF0::native_outputs`] on one session, over already-padded audio.
fn outputs_on(session: &mut Session, audio: &[f32]) -> Result<(Vec<f32>, Vec<f32>), Error> {
    let length = audio.len();
    let input = Tensor::from_array(([1_i64, length as i64], audio.to_vec().into_boxed_slice()))
        .map_err(session::ort_error)?;
    let outputs = session
        .run(ort::inputs![input])
        .map_err(|error| Error::Ort(format!("SwiftF0 inference failed: {error}")))?;

    let (pitch_shape, pitch) = outputs
        .get(PITCH_OUTPUT)
        .ok_or_else(|| Error::Ort(format!("SwiftF0 has no `{PITCH_OUTPUT}` output")))?
        .try_extract_tensor::<f32>()
        .map_err(|error| {
            Error::Ort(format!(
                "SwiftF0 `{PITCH_OUTPUT}` is not an f32 tensor: {error}"
            ))
        })?;
    let (confidence_shape, confidence) = outputs
        .get(CONFIDENCE_OUTPUT)
        .ok_or_else(|| Error::Ort(format!("SwiftF0 has no `{CONFIDENCE_OUTPUT}` output")))?
        .try_extract_tensor::<f32>()
        .map_err(|error| {
            Error::Ort(format!(
                "SwiftF0 `{CONFIDENCE_OUTPUT}` is not an f32 tensor: {error}"
            ))
        })?;

    let frames = validate_output_shape(PITCH_OUTPUT, pitch_shape)?;
    validate_output_shape(CONFIDENCE_OUTPUT, confidence_shape)?;
    if pitch.len() != confidence.len() || confidence.len() != frames {
        return Err(Error::Ort(format!(
            "SwiftF0 pitch/confidence length mismatch: {} vs {}",
            pitch.len(),
            confidence.len()
        )));
    }
    Ok((pitch.to_vec(), confidence.to_vec()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_bundled_model_is_the_pinned_size() {
        // The full SHA-256 provenance check lives in the integration tests,
        // where it can use a hashing crate; the size is the cheap guard here.
        assert_eq!(BUNDLED_MODEL.len(), 397_987);
    }

    #[test]
    fn resolve_source_always_yields_a_source() {
        // A file override or the bundled fallback, never an error: the modern
        // build ships a working SwiftF0 with no download.
        let source = resolve_source();
        assert_eq!(source.file_name(), SWIFTF0_FILE_NAME);
    }
}
