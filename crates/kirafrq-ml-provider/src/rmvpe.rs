//! RMVPE over ONNX Runtime: the salience decoder, the #53 policy wiring and
//! the lazily-built session. Every `ort` call in the workspace lives here.
//!
//! Contract (verified against `rvc/lib/rmvpe.py`, `pitch-core-onnx::rmvpe`
//! and the local `rmvpe.onnx` in #36/#38):
//!
//! - input: log-mel `[1, 128, T]` at 16 kHz ([`crate::mel`]), `T` padded up
//!   to a multiple of 32.
//! - output: salience `[1, T, 360]`; f0 is the salience-weighted mean over
//!   `+/-4` bins around the argmax, confidence is the peak salience.

use std::path::PathBuf;
use std::sync::Mutex;

use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;

use crate::grid::{NativeContour, RMVPE_CONTRACT, Track, map_to_table_grid};
use crate::mel::Frontend;
use crate::policy::UvPolicy;
use crate::resample::resample_to_model_rate;
use crate::{Error, ProgressObserver};

/// Salience bins on RMVPE's 20-cent grid.
pub const N_BINS: usize = 360;
const CENTS_PER_BIN: f64 = 20.0;
/// Bin 0's pitch in cents above 10 Hz (CREPE's grid, shared by RMVPE).
const CENTS_OFFSET: f64 = 1_997.379_408_437_619;
const DECODER_HALF_WINDOW: usize = 4;

/// Progress ticks per estimate: resample, inference, decode, map.
const PROGRESS_TICKS: usize = 4;

/// The RMVPE estimator: one session per process, built lazily behind a mutex.
pub struct Rmvpe {
    model_path: PathBuf,
    policy: UvPolicy,
    /// `intra_op_num_threads`; `0` leaves ONNX Runtime's default (physical
    /// cores). Clamped against the machine when the session is built.
    jobs: usize,
    session: Mutex<Option<Session>>,
}

impl Rmvpe {
    /// The estimator for `model_path` with the #53 policy; `jobs` follows the
    /// run's single "cores" knob (0 = ONNX Runtime default).
    pub fn new(model_path: PathBuf, policy: UvPolicy, jobs: usize) -> Self {
        Self {
            model_path,
            policy,
            jobs,
            session: Mutex::new(None),
        }
    }

    /// Estimate f0 for mono 44.1 kHz `samples` and map it onto the table grid
    /// (#47): resample to 16 kHz, run RMVPE, apply the #53 policy on the
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
        if samples.len() < crate::grid::TABLE_HOP_SAMPLES {
            return Ok(map_to_table_grid(
                &NativeContour {
                    contract: RMVPE_CONTRACT,
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
            contract: RMVPE_CONTRACT,
            f0_hz: native_f0,
        };
        let track = map_to_table_grid(&contour, samples.len());
        report(PROGRESS_TICKS);
        Ok(track)
    }

    /// One full inference: mel frontend, session run, decoder with the #53
    /// policy on the native grid.
    fn infer(&self, audio_16k: &[f32]) -> Result<Vec<f64>, Error> {
        let mut frontend = Frontend::new();
        let (mel, frames) = frontend.mel(audio_16k);
        let padded = frontend.pad_to_multiple(&mel, frames);
        let salience = self.run_session(&padded, frames)?;
        Ok(decode_salience(&salience, frames, &self.policy))
    }

    /// The policy-applied f0 on the model's native grid, for local
    /// verification probes against the Python reference. Not part of the
    /// pipeline surface.
    #[doc(hidden)]
    pub fn native_estimate(&self, audio_16k: &[f32]) -> Result<Vec<f64>, Error> {
        self.infer(audio_16k)
    }

    /// Run the session over a padded `[1, 128, T_pad]` log-mel input and
    /// return the first `frames` rows of the `[1, T_pad, 360]` salience.
    fn run_session(&self, mel: &[f32], frames: usize) -> Result<Vec<f32>, Error> {
        let mut guard = self
            .session
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if guard.is_none() {
            *guard = Some(self.build_session()?);
        }
        let session = guard.as_mut().expect("just built");

        let padded = mel.len() / crate::mel::N_MELS;
        let input = Tensor::from_array((
            [1_i64, crate::mel::N_MELS as i64, padded as i64],
            mel.to_vec().into_boxed_slice(),
        ))
        .map_err(ort_error)?;
        let outputs = session
            .run(ort::inputs![input])
            .map_err(|error| Error::Ort(format!("RMVPE inference failed: {error}")))?;
        let (shape, salience) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|error| Error::Ort(format!("RMVPE output is not an f32 tensor: {error}")))?;

        let shape: Vec<usize> = shape.iter().map(|&dim| dim as usize).collect();
        if shape.len() != 3 || shape[0] != 1 || shape[2] != N_BINS || shape[1] < frames {
            return Err(Error::Ort(format!(
                "RMVPE output shape {shape:?} is not [1, >= {frames}, {N_BINS}]"
            )));
        }
        Ok(salience[..frames * N_BINS].to_vec())
    }

    /// Build the session: static ONNX Runtime, graph optimization at the
    /// default level, `inter_op = 1`, `jobs` into the intra-op pool.
    fn build_session(&self) -> Result<Session, Error> {
        let mut builder = Session::builder().map_err(ort_error)?;
        builder = builder
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(ort_error)?;
        builder = builder.with_inter_threads(1).map_err(ort_error)?;
        if self.jobs > 0 {
            builder = builder
                .with_intra_threads(intra_op_threads(self.jobs))
                .map_err(ort_error)?;
        }
        builder.commit_from_file(&self.model_path).map_err(|error| {
            Error::Ort(format!(
                "cannot load {}: {error}",
                self.model_path.display()
            ))
        })
    }
}

/// `jobs` clamped to something the machine can use: at least one thread, at
/// most the available parallelism (an ORT default when `jobs` is 0).
fn intra_op_threads(jobs: usize) -> usize {
    let available = std::thread::available_parallelism()
        .map(|cores| cores.get())
        .unwrap_or(1);
    jobs.clamp(1, available.max(1))
}

fn ort_error(error: impl std::fmt::Display) -> Error {
    Error::Ort(error.to_string())
}

/// Decode the salience of the first `frames` native frames into f0 with the
/// #53 policy applied: weighted mean over `+/-4` bins around the argmax, the
/// peak as confidence, `0.0` when unvoiced.
pub fn decode_salience(salience: &[f32], frames: usize, policy: &UvPolicy) -> Vec<f64> {
    let mut f0 = Vec::with_capacity(frames);
    for frame in 0..frames {
        let row = &salience[frame * N_BINS..(frame + 1) * N_BINS];
        let (peak, confidence) = row.iter().copied().enumerate().fold(
            (0usize, f32::NEG_INFINITY),
            |best, (bin, value)| {
                if value > best.1 { (bin, value) } else { best }
            },
        );

        let lo = peak.saturating_sub(DECODER_HALF_WINDOW);
        let hi = (peak + DECODER_HALF_WINDOW).min(N_BINS - 1);
        let mut weighted_cents = 0.0_f64;
        let mut total = 0.0_f64;
        for (bin, &value) in row.iter().enumerate().take(hi + 1).skip(lo) {
            let weight = f64::from(value.max(0.0));
            weighted_cents += weight * (bin as f64 * CENTS_PER_BIN + CENTS_OFFSET);
            total += weight;
        }
        let cents = if total > 0.0 {
            weighted_cents / total
        } else {
            peak as f64 * CENTS_PER_BIN + CENTS_OFFSET
        };
        let pitch_hz = 10.0 * 2.0_f64.powf(cents / 1200.0);
        f0.push(policy.apply(pitch_hz, f64::from(confidence)));
    }
    f0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intra_op_threads_clamp_to_the_machine() {
        let available = std::thread::available_parallelism()
            .map(|cores| cores.get())
            .unwrap_or(1);
        assert_eq!(intra_op_threads(1), 1);
        assert_eq!(intra_op_threads(available), available);
        assert_eq!(intra_op_threads(usize::MAX), available);
    }

    #[test]
    fn the_decoder_reads_the_peak_bin_through_the_policy() {
        // A salience row with its peak in bin 228 (~441.5 Hz): voiced with
        // confidence 1.0.
        let mut row = vec![0.0_f32; N_BINS];
        row[228] = 1.0;
        let f0 = decode_salience(&row, 1, &UvPolicy::rmvpe(None));
        assert!((f0[0] - 441.5).abs() < 1.0, "{:?}", f0);

        // The same row with a low peak reads unvoiced.
        row[228] = 0.01;
        let f0 = decode_salience(&row, 1, &UvPolicy::rmvpe(None));
        assert_eq!(f0, vec![0.0]);
    }

    #[test]
    fn the_decoder_never_emits_a_value_outside_the_range() {
        // A peak in bin 0 (~31.7 Hz) is below the floor and must be dropped,
        // not clamped.
        let mut row = vec![0.0_f32; N_BINS];
        row[0] = 1.0;
        assert_eq!(decode_salience(&row, 1, &UvPolicy::rmvpe(None)), vec![0.0]);

        // Bin 359 (~3951 Hz) is above the ceiling.
        let mut row = vec![0.0_f32; N_BINS];
        row[359] = 1.0;
        assert_eq!(decode_salience(&row, 1, &UvPolicy::rmvpe(None)), vec![0.0]);
    }

    #[test]
    fn an_all_zero_row_is_unvoiced_and_finite() {
        let f0 = decode_salience(&vec![0.0_f32; N_BINS], 1, &UvPolicy::rmvpe(None));
        assert_eq!(f0, vec![0.0]);
    }
}
