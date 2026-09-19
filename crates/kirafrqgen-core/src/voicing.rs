//! The energy voicing gate (#54): the tuned path's post-pass over an f0 track,
//! forcing frames quieter than a share of the file's voiced-frame amplitude
//! unvoiced.

use crate::F0Track;
use crate::table::amplitudes;

/// Force frames below `ratio` of the file's p90 voiced-frame amplitude
/// unvoiced (#46/#54). The reference is the p90 of the frq amplitudes of the
/// frames the track calls voiced; no voiced frames or a zero p90 is a no-op,
/// so a silence-only file can never produce NaN or a degenerate table.
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

/// The p90 of the amplitudes of the voiced frames (`f0 > 0`), `None` when no
/// frame is voiced; `NaN` frames are not voiced (#8 reads them unvoiced).
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
/// voiced set interpolates between order statistics instead of snapping.
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

    const HOP: usize = 256;

    fn track(f0_hz: &[f64]) -> F0Track {
        F0Track {
            frame_period_ms: HOP as f64 / 44_100.0 * 1000.0,
            temporal_positions: (0..f0_hz.len())
                .map(|index| index as f64 * HOP as f64 / 44_100.0)
                .collect(),
            f0_hz: f0_hz.to_vec(),
        }
    }

    /// Constant samples per `(value, length)` window; frame `i` reads window
    /// `i` (the last window may be short).
    fn samples(windows: &[(f64, usize)]) -> Vec<f64> {
        windows
            .iter()
            .flat_map(|(value, length)| vec![*value; *length])
            .collect()
    }

    /// `32768 * value` per full window, so `0.25` reads `8192.0`.
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
    fn gate_never_emits_nan() {
        let samples = samples(&[(0.001, HOP), (0.25, HOP), (0.25, HOP)]);
        let mut track = track(&[f64::NAN, 220.0, 330.0]);

        apply_energy_gate(&samples, &mut track, 0.05);

        assert!(track.f0_hz.iter().all(|value| value.is_finite()));
    }
}
