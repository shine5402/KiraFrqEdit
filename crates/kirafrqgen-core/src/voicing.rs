//! The tuned WORLD path's voicing gates (#54/#64): post-passes over an f0
//! track that force frames unvoiced. The energy gate drops frames quieter than
//! a share of the file's voiced-frame amplitude; the aperiodicity gate drops
//! Harvest frames with a low raw D4C LoveTrain statistic.

use crate::F0Track;
use crate::table::amplitudes;

/// Force frames below `ratio` of the file's p90 voiced-frame amplitude
/// unvoiced (#46/#54). A no-op when the track has no voiced frame or the p90
/// is zero, so a silence-only file cannot turn into a degenerate table.
pub(crate) fn apply_energy_gate(samples: &[f64], track: &mut F0Track, ratio: f64) {
    let amplitudes = amplitudes(samples);
    let Some(reference) = voiced_p90(&amplitudes, &track.f0_hz) else {
        return;
    };
    if reference <= 0.0 {
        return;
    }
    let threshold = reference * ratio;
    for (f0_hz, amplitude) in track.f0_hz.iter_mut().zip(&amplitudes) {
        if *amplitude < threshold {
            *f0_hz = 0.0;
        }
    }
}

/// Force voiced frames whose raw D4C LoveTrain statistic is below `threshold`
/// unvoiced (#55/#64). The statistics are per track frame; a frame without one
/// (a short slice) is left untouched. Only ever writes `0.0`, so the gate can
/// never add voicing.
pub(crate) fn apply_aperiodicity_gate(track: &mut F0Track, statistic: &[f64], threshold: f64) {
    for (f0_hz, &value) in track.f0_hz.iter_mut().zip(statistic) {
        if *f0_hz > 0.0 && value < threshold {
            *f0_hz = 0.0;
        }
    }
}

/// Whether the track still carries a frame the gates could act on: a finite,
/// positive f0. Silence-only / no-voiced tracks skip the D4C pass (#64).
pub(crate) fn has_voiced(track: &F0Track) -> bool {
    track
        .f0_hz
        .iter()
        .any(|value| value.is_finite() && *value > 0.0)
}

/// The p90 of the voiced frames' amplitudes: `f0 > 0`, with `NaN` excluded
/// (#8 reads non-finite f0 as unvoiced).
fn voiced_p90(amplitudes: &[f64], f0_hz: &[f64]) -> Option<f64> {
    let mut voiced: Vec<f64> = f0_hz
        .iter()
        .zip(amplitudes)
        .filter_map(|(&f0_hz, &amplitude)| (f0_hz.is_finite() && f0_hz > 0.0).then_some(amplitude))
        .collect();
    if voiced.is_empty() {
        return None;
    }
    voiced.sort_by(f64::total_cmp);
    Some(percentile(&voiced, 0.9))
}

/// Linear-interpolated percentile (the `np.percentile` default), so a small
/// voiced set interpolates between order statistics.
fn percentile(sorted: &[f64], fraction: f64) -> f64 {
    let position = (sorted.len() - 1) as f64 * fraction;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    let weight = position - lower as f64;
    sorted[lower] + weight * (sorted[upper] - sorted[lower])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::table::{HOP_SAMPLES, frame_period_ms};

    const HOP: usize = HOP_SAMPLES as usize;

    fn track(f0_hz: &[f64]) -> F0Track {
        let period = frame_period_ms();
        F0Track {
            frame_period_ms: period,
            temporal_positions: (0..f0_hz.len())
                .map(|index| index as f64 * period / 1000.0)
                .collect(),
            f0_hz: f0_hz.to_vec(),
        }
    }

    /// Constant samples per `(value, length)` window; frame `i` reads window
    /// `i` (the last window may be short). A full window of `value` reads a frq
    /// amplitude of `32768 * value`, so `0.25` reads `8192.0`.
    fn samples(windows: &[(f64, usize)]) -> Vec<f64> {
        windows
            .iter()
            .flat_map(|(value, length)| vec![*value; *length])
            .collect()
    }

    #[test]
    fn gate_zeroes_frames_below_five_percent_of_the_voiced_p90() {
        let samples = samples(&[(0.25, HOP), (0.001, HOP), (0.25, HOP), (0.25, 128)]);
        let mut track = track(&[110.0, 220.0, 330.0, 440.0]);

        apply_energy_gate(&samples, &mut track, 0.05);

        assert_eq!(
            track.f0_hz,
            [110.0, 0.0, 330.0, 440.0],
            "the 32.768-amplitude frame is under 5% of the 8192 p90"
        );
    }

    #[test]
    fn gate_keeps_a_frame_exactly_at_the_threshold() {
        let samples = samples(&[(0.5, HOP), (0.25, HOP), (0.5, HOP), (0.5, 128)]);
        let mut track = track(&[110.0, 220.0, 330.0, 440.0]);

        apply_energy_gate(&samples, &mut track, 0.5);

        assert_eq!(
            track.f0_hz,
            [110.0, 220.0, 330.0, 440.0],
            "8192 is not below the 16384 * 0.5 threshold"
        );
    }

    #[test]
    fn the_reference_ignores_unvoiced_frames() {
        let quiet = 100.0 / 32_768.0;
        let samples = samples(&[(quiet, HOP), (quiet, HOP), (quiet, HOP), (1.0, 128)]);
        let mut track = track(&[100.0, 100.0, 100.0, 0.0]);

        apply_energy_gate(&samples, &mut track, 0.05);

        assert_eq!(
            track.f0_hz,
            [100.0, 100.0, 100.0, 0.0],
            "the loud unvoiced frame must not raise the p90"
        );
    }

    #[test]
    fn gate_is_a_no_op_without_voiced_frames() {
        let samples = samples(&[(0.25, HOP), (0.001, HOP)]);
        let mut track = track(&[0.0, 0.0]);

        apply_energy_gate(&samples, &mut track, 0.05);

        assert_eq!(track.f0_hz, [0.0, 0.0]);
    }

    #[test]
    fn gate_is_a_no_op_when_the_voiced_p90_is_zero() {
        let samples = samples(&[(0.0, HOP), (0.0, HOP)]);
        let mut track = track(&[220.0, 220.0]);

        apply_energy_gate(&samples, &mut track, 0.05);

        assert_eq!(track.f0_hz, [220.0, 220.0], "p90 = 0 skips the gate");
    }

    #[test]
    fn gate_never_introduces_nan() {
        let samples = samples(&[(0.001, HOP), (0.25, HOP), (0.25, HOP)]);
        let mut track = track(&[f64::NAN, 220.0, 330.0]);

        apply_energy_gate(&samples, &mut track, 0.05);

        assert!(
            track.f0_hz.iter().all(|value| value.is_finite()),
            "the gate only ever writes 0.0"
        );
    }

    #[test]
    fn aperiodicity_gate_zeroes_frames_below_the_threshold() {
        let mut track = track(&[110.0, 220.0, 330.0, 440.0]);
        let statistic = [0.9, 0.84, 0.85, 0.1];

        apply_aperiodicity_gate(&mut track, &statistic, 0.85);

        assert_eq!(
            track.f0_hz,
            [110.0, 0.0, 330.0, 0.0],
            "0.84 is below the threshold, 0.85 is not (strict less-than)"
        );
    }

    #[test]
    fn aperiodicity_gate_never_adds_voicing() {
        let mut track = track(&[0.0, 0.0, 0.0]);
        let statistic = [0.0, 0.99, 0.5];

        apply_aperiodicity_gate(&mut track, &statistic, 0.85);

        assert_eq!(track.f0_hz, [0.0, 0.0, 0.0], "only ever writes 0.0");
    }

    #[test]
    fn aperiodicity_gate_leaves_frames_past_a_short_statistic_untouched() {
        let mut track = track(&[110.0, 220.0, 330.0]);
        let statistic = [0.1, 0.9];

        apply_aperiodicity_gate(&mut track, &statistic, 0.85);

        assert_eq!(
            track.f0_hz,
            [0.0, 220.0, 330.0],
            "a frame without a statistic is not gated"
        );
    }

    #[test]
    fn aperiodicity_gate_is_a_no_op_without_frames() {
        let mut track = track(&[]);

        apply_aperiodicity_gate(&mut track, &[], 0.85);

        assert!(track.f0_hz.is_empty());
    }

    #[test]
    fn has_voiced_reads_finite_positive_frames() {
        assert!(has_voiced(&track(&[0.0, 220.0])));
        assert!(!has_voiced(&track(&[0.0, f64::NAN])));
        assert!(!has_voiced(&track(&[-1.0, 0.0])));
        assert!(!has_voiced(&track(&[])));
    }
}
