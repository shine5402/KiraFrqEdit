//! The ML path's input resample: the pipeline's mono 44.1 kHz audio to the
//! model's expected rate with rubato `Fft`, the same offline settings #11
//! fixed for the decode path (`process_all`, `FixedSync::Both`) and #47
//! reused. The double resample for stray-rate sources is accepted.

use rubato::Resampler;
use rubato::audioadapter_buffers::direct::InterleavedSlice;

use crate::Error;

/// Chunk hint for [`rubato::Fft::new`] (the offline-batch settings from #11).
const CHUNK_HINT: usize = 2048;

/// Resample `samples` from `input_rate` to `output_rate`.
///
/// Equal rates return a copy unchanged. The output holds exactly
/// `ceil(ratio * len)` samples, matching the decode path's convention.
pub fn resample_to_model_rate(
    samples: &[f64],
    input_rate: u32,
    output_rate: u32,
) -> Result<Vec<f64>, Error> {
    if samples.is_empty() || input_rate == output_rate {
        return Ok(samples.to_vec());
    }

    let failure = |error: &dyn std::fmt::Display| {
        Error::Resample(format!("{input_rate} Hz to {output_rate} Hz: {error}"))
    };

    let mut resampler = rubato::Fft::<f64>::new(
        input_rate as usize,
        output_rate as usize,
        CHUNK_HINT,
        1,
        rubato::FixedSync::Both,
    )
    .map_err(|error| failure(&error))?;

    // `process_all` only trims its startup delay after a full internal chunk;
    // a clip shorter than one chunk would come back untrimmed. Pad it past
    // one chunk so the trim runs, then cut the zero tail (#11).
    let original_len = samples.len();
    let padded_len = original_len.max(resampler.input_frames_next() + 1);
    let mut padded = samples.to_vec();
    padded.resize(padded_len, 0.0);

    let input = InterleavedSlice::new(&padded, 1, padded_len).map_err(|error| failure(&error))?;
    let output = resampler
        .process_all(&input, padded_len, None)
        .map_err(|error| failure(&error))?;

    let ratio = f64::from(output_rate) / f64::from(input_rate);
    let mut samples = output.take_data();
    samples.truncate((ratio * original_len as f64).ceil() as usize);
    Ok(samples)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Correlation energy of `samples` at `hz` (a one-bin DFT).
    fn tone_energy(samples: &[f64], rate: u32, hz: f64) -> f64 {
        let mut re = 0.0;
        let mut im = 0.0;
        for (index, &sample) in samples.iter().enumerate() {
            let phase = std::f64::consts::TAU * hz * index as f64 / rate as f64;
            re += sample * phase.cos();
            im += sample * phase.sin();
        }
        (re * re + im * im).sqrt() / samples.len() as f64
    }

    fn sine(len: usize, rate: u32, hz: f64) -> Vec<f64> {
        (0..len)
            .map(|index| (std::f64::consts::TAU * hz * index as f64 / rate as f64).sin())
            .collect()
    }

    #[test]
    fn equal_rates_return_the_input_unchanged() {
        let input: Vec<f64> = (0..1000).map(|index| index as f64 * 0.001).collect();
        let output = resample_to_model_rate(&input, 44_100, 44_100).unwrap();
        assert_eq!(output, input);
    }

    #[test]
    fn empty_input_stays_empty() {
        assert_eq!(
            resample_to_model_rate(&[], 44_100, 16_000).unwrap(),
            Vec::<f64>::new()
        );
    }

    #[test]
    fn output_length_is_the_ceil_of_the_ratio() {
        for len in [1, 255, 256, 1000, 13_824, 44_100, 147_000] {
            let input = vec![0.0; len];
            let output = resample_to_model_rate(&input, 44_100, 16_000).unwrap();
            let expected = (16_000.0 / 44_100.0 * len as f64).ceil() as usize;
            assert_eq!(output.len(), expected, "len = {len}");
        }
    }

    #[test]
    fn a_tone_keeps_its_frequency_through_the_resample() {
        let input = sine(22_050, 44_100, 440.0);
        let output = resample_to_model_rate(&input, 44_100, 16_000).unwrap();
        assert_eq!(output.len(), 8000);

        let at_440 = tone_energy(&output, 16_000, 440.0);
        let at_300 = tone_energy(&output, 16_000, 300.0);
        let at_600 = tone_energy(&output, 16_000, 600.0);
        assert!(at_440 > 10.0 * at_300, "440 vs 300: {at_440} vs {at_300}");
        assert!(at_440 > 10.0 * at_600, "440 vs 600: {at_440} vs {at_600}");
    }
}
