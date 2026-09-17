//! Building the neutral [`FrequencyTable`] from decoded samples and an f0
//! track — #8's numeric policy.

use frq_core::FrequencyTable;

use crate::F0Track;

/// Every UTAU table format shares the 44.1 kHz, 256-sample hop (#7).
pub(crate) const SAMPLE_RATE: u32 = 44_100;
pub(crate) const HOP_SAMPLES: u32 = 256;

/// The WORLD frame period that lands estimator frames exactly on table
/// frames.
pub(crate) fn frame_period_ms() -> f64 {
    f64::from(HOP_SAMPLES) / f64::from(SAMPLE_RATE) * 1000.0
}

/// The neutral table for a decoded wav on the `N = floor(L/hop) + 1` frame
/// grid (#8): frames past the grid are dropped, missing ones read unvoiced,
/// and the trailing incomplete window is forced unvoiced.
pub(crate) fn build_table(
    samples: &[f64],
    track: &F0Track,
    want_amplitude: bool,
) -> FrequencyTable {
    debug_assert!(
        !samples.is_empty(),
        "zero-length wavs are skipped before analysis"
    );
    let hop = HOP_SAMPLES as usize;
    let frames = samples.len() / hop + 1;

    let mut f0_hz = vec![0.0; frames];
    for (frame, &value) in f0_hz.iter_mut().zip(track.f0_hz.iter()) {
        if value.is_finite() && value > 0.0 {
            *frame = value;
        }
    }
    // `N = floor(L/hop) + 1` makes `N*hop > L` for every positive `L`, so the
    // trailing frame's window is incomplete (or empty when `L % hop == 0`).
    f0_hz[frames - 1] = 0.0;

    let mut voiced_sum = 0.0;
    let mut voiced_count = 0usize;
    for &value in &f0_hz {
        if value > 0.0 {
            voiced_sum += value;
            voiced_count += 1;
        }
    }
    let key_hz = if voiced_count == 0 {
        0.0
    } else {
        voiced_sum / voiced_count as f64
    };

    let amplitude = want_amplitude.then(|| amplitudes(samples));

    FrequencyTable {
        sample_rate: SAMPLE_RATE,
        hop_samples: HOP_SAMPLES,
        f0_hz,
        amplitude,
        key_hz,
    }
}

/// `2^15 * mean(|x|)` per frame, the divisor clipping to the partial last
/// window; the empty trailing frame of an `L % hop == 0` wav is `0.0` (#8).
fn amplitudes(samples: &[f64]) -> Vec<f64> {
    let hop = HOP_SAMPLES as usize;
    let frames = samples.len() / hop + 1;
    (0..frames)
        .map(|frame| {
            let lo = frame * hop;
            let hi = (lo + hop).min(samples.len());
            if hi <= lo {
                return 0.0;
            }
            let sum: f64 = samples[lo..hi].iter().map(|sample| sample.abs()).sum();
            32_768.0 / (hi - lo) as f64 * sum
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn track(f0_hz: Vec<f64>) -> F0Track {
        let period = frame_period_ms();
        F0Track {
            frame_period_ms: period,
            temporal_positions: (0..f0_hz.len())
                .map(|i| i as f64 * period / 1000.0)
                .collect(),
            f0_hz,
        }
    }

    #[test]
    fn grid_is_floor_length_over_hop_plus_one() {
        for (length, frames) in [(1, 1), (255, 1), (256, 2), (1000, 4), (1024, 5)] {
            let samples = vec![0.0; length];
            let source: Vec<f64> = (1..=frames as u32).map(f64::from).collect();
            let table = build_table(&samples, &track(source.clone()), false);
            assert_eq!(table.f0_hz.len(), frames, "L = {length}");
            assert_eq!(table.sample_rate, 44_100);
            assert_eq!(table.hop_samples, 256);

            // Frame N-1 is always forced unvoiced; the earlier frames keep the
            // track's values.
            for (index, (got, want)) in table.f0_hz[..frames - 1].iter().zip(&source).enumerate() {
                assert_eq!(got, want, "L = {length}, i = {index}");
            }
            assert_eq!(table.f0_hz[frames - 1], 0.0, "L = {length}, trailing frame");
        }
    }

    #[test]
    fn pads_short_tracks_and_truncates_long_ones() {
        let samples = vec![0.0; 1000];
        let short = build_table(&samples, &track(vec![100.0, 200.0]), false);
        assert_eq!(short.f0_hz, [100.0, 200.0, 0.0, 0.0]);

        let long = build_table(
            &samples,
            &track(vec![100.0, 200.0, 300.0, 400.0, 500.0, 600.0]),
            false,
        );
        assert_eq!(long.f0_hz, [100.0, 200.0, 300.0, 0.0]);
    }

    #[test]
    fn non_finite_and_negative_f0_read_unvoiced() {
        let samples = vec![0.0; 512];
        let table = build_table(
            &samples,
            &track(vec![f64::NAN, -10.0, f64::INFINITY, 220.0]),
            false,
        );
        assert_eq!(table.f0_hz, [0.0, 0.0, 0.0], "trailing rule included");
    }

    #[test]
    fn key_is_the_unrounded_voiced_mean() {
        let samples = vec![0.0; 1000];
        let table = build_table(&samples, &track(vec![100.0, 200.0, 300.0, 400.0]), false);
        assert_eq!(table.f0_hz, [100.0, 200.0, 300.0, 0.0]);
        assert_eq!(table.key_hz, 200.0);

        let unrounded = build_table(&samples, &track(vec![100.0, 101.0, 0.0, 0.0]), false);
        assert_eq!(unrounded.key_hz, 100.5);
    }

    #[test]
    fn key_is_zero_when_nothing_is_voiced() {
        let samples = vec![0.0; 1000];
        let table = build_table(&samples, &track(vec![0.0; 4]), false);
        assert_eq!(table.key_hz, 0.0);
        assert_eq!(table.f0_hz, [0.0; 4]);
    }

    #[test]
    fn amplitude_is_mean_abs_scaled_over_each_window() {
        let samples = vec![0.25; 1000];
        let table = build_table(&samples, &track(vec![100.0; 4]), true);
        let amplitude = table.amplitude.clone().unwrap();
        assert_eq!(amplitude, [8192.0; 4], "partial window keeps the mean");

        let full = build_table(&vec![0.25; 1024], &track(vec![100.0; 5]), true);
        let amplitude = full.amplitude.clone().unwrap();
        assert_eq!(
            amplitude,
            [8192.0, 8192.0, 8192.0, 8192.0, 0.0],
            "the empty trailing window is 0.0"
        );
    }

    #[test]
    fn amplitude_is_only_computed_when_requested() {
        let samples = vec![0.5; 512];
        let table = build_table(&samples, &track(vec![100.0, 0.0, 0.0]), false);
        assert!(table.amplitude.is_none());
    }
}
