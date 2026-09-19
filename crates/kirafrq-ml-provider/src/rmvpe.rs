//! RMVPE over ONNX Runtime: the salience decoder, the #53 policy wiring and
//! the lazily-built session.
//!
//! Contract (verified against `rvc/lib/rmvpe.py`, `pitch-core-onnx::rmvpe`
//! and the local `rmvpe.onnx` in #36/#38):
//!
//! - input: log-mel `[1, 128, T]` at 16 kHz ([`crate::mel`]), `T` padded up
//!   to a multiple of 32.
//! - output: salience `[1, T, 360]`; f0 is the salience-weighted mean over
//!   `+/-4` bins around the argmax, confidence is the peak salience.

use ort::value::Tensor;

use crate::grid::{NativeContour, RMVPE_CONTRACT, Track, map_to_table_grid};
use crate::mel::Frontend;
use crate::model::ModelSource;
use crate::policy::UvPolicy;
use crate::resample::resample_to_model_rate;
use crate::session;
use crate::{Error, ProgressObserver};

/// Salience bins on RMVPE's 20-cent grid.
pub const N_BINS: usize = 360;
const CENTS_PER_BIN: f64 = 20.0;
/// Bin 0's pitch in cents above 10 Hz (CREPE's grid, shared by RMVPE).
const CENTS_OFFSET: f64 = 1_997.379_408_437_619;
const DECODER_HALF_WINDOW: usize = 4;

/// Progress ticks per estimate: resample, inference, decode, map.
const PROGRESS_TICKS: usize = 4;

/// The RMVPE estimator. The ONNX Runtime session is a process-wide,
/// lazily-built singleton behind a mutex, so every run in one process reuses
/// the loaded model.
pub struct Rmvpe {
    source: ModelSource,
    policy: UvPolicy,
    /// `intra_op_num_threads`; `0` leaves ONNX Runtime's default (physical
    /// cores). Clamped against the machine when the session is built.
    jobs: usize,
}

impl Rmvpe {
    /// The estimator for `source` with the #53 policy; `jobs` follows the
    /// run's single "cores" knob (0 = ONNX Runtime default).
    pub fn new(source: impl Into<ModelSource>, policy: UvPolicy, jobs: usize) -> Self {
        Self {
            source: source.into(),
            policy,
            jobs,
        }
    }

    /// Build the process session now if it is not built yet. A model that
    /// cannot be loaded then fails at configuration time (#49) instead of on
    /// the first file.
    pub fn ensure_session(&self) -> Result<(), Error> {
        session::with(&self.source, self.jobs, |_| Ok(()))
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

    /// Run the process session over a padded `[1, 128, T_pad]` log-mel input
    /// and return the first `frames` rows of the `[1, T_pad, 360]` salience.
    fn run_session(&self, mel: &[f32], frames: usize) -> Result<Vec<f32>, Error> {
        session::with(&self.source, self.jobs, |session| {
            let padded = mel.len() / crate::mel::N_MELS;
            let input = Tensor::from_array((
                [1_i64, crate::mel::N_MELS as i64, padded as i64],
                mel.to_vec().into_boxed_slice(),
            ))
            .map_err(session::ort_error)?;
            let outputs = session
                .run(ort::inputs![input])
                .map_err(|error| Error::Ort(format!("RMVPE inference failed: {error}")))?;
            let (shape, salience) = outputs[0].try_extract_tensor::<f32>().map_err(|error| {
                Error::Ort(format!("RMVPE output is not an f32 tensor: {error}"))
            })?;

            let shape: Vec<usize> = shape.iter().map(|&dim| dim as usize).collect();
            if shape.len() != 3 || shape[0] != 1 || shape[2] != N_BINS || shape[1] < frames {
                return Err(Error::Ort(format!(
                    "RMVPE output shape {shape:?} is not [1, >= {frames}, {N_BINS}]"
                )));
            }
            Ok(salience[..frames * N_BINS].to_vec())
        })
    }
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

    #[test]
    fn a_single_voiced_native_frame_never_voices_a_table_frame() {
        // #53 criterion 6, end to end through the decoder and the #47
        // mapper: frame 1 is voiced, frames 0 and 2 are not, so no table
        // frame's bracket is fully voiced.
        let mut salience = vec![0.0_f32; 3 * N_BINS];
        salience[N_BINS + 228] = 1.0;
        let native = decode_salience(&salience, 3, &UvPolicy::rmvpe(None));
        assert!(native[1] > 0.0, "the middle native frame is voiced");
        let track = map_to_table_grid(
            &NativeContour {
                contract: RMVPE_CONTRACT,
                f0_hz: native,
            },
            2048,
        );
        assert!(
            track.f0_hz.iter().all(|&value| value == 0.0),
            "{:?}",
            track.f0_hz
        );
    }

    #[test]
    fn an_all_below_threshold_stream_is_all_unvoiced() {
        // #53 criterion 4 at the provider level: with every peak under the
        // threshold the decoded contour and the mapped track are all `0.0`.
        let rows = 4;
        let mut salience = vec![0.0_f32; rows * N_BINS];
        for row in 0..rows {
            salience[row * N_BINS + 228] = 0.01;
        }
        let policy = UvPolicy::rmvpe(Some(0.03));
        let native = decode_salience(&salience, rows, &policy);
        assert_eq!(native, vec![0.0; rows]);

        let track = map_to_table_grid(
            &NativeContour {
                contract: RMVPE_CONTRACT,
                f0_hz: native,
            },
            2048,
        );
        assert_eq!(track.f0_hz, vec![0.0; 9]);
    }
}
