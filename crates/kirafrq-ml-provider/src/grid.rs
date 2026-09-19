//! The #47 native-contour -> table-grid mapping: pure geometry, no model.
//!
//! The estimator emits exactly `N = floor(L / 256) + 1` frames at
//! `t = i * 256 / 44100` (5.805 ms), so `build_table`'s 1:1 zip is exact and
//! never leans on pad/truncate. Native f0 is interpolated linearly in Hz, and
//! only across consecutive-voiced native frames: a table frame is voiced iff
//! **both** bracketing native frames are voiced. The trailing-frame rule and
//! the key frequency are the table builder's, untouched (#8).

/// Every UTAU table format shares the 44.1 kHz, 256-sample hop (#7).
pub const TABLE_SAMPLE_RATE: u32 = 44_100;
/// Frames are 256 samples apart at 44.1 kHz, the 5.805 ms table grid.
pub const TABLE_HOP_SAMPLES: usize = 256;

/// The exact frame count #8 guarantees for a `L`-sample 44.1 kHz wav:
/// `N = floor(L / 256) + 1`.
pub fn table_frame_count(input_len_44100: usize) -> usize {
    input_len_44100 / TABLE_HOP_SAMPLES + 1
}

/// One model's native time base (#47): native frame `k` sits at
/// `t = origin_s + k * hop_samples / sample_rate`. `origin_s` carries a
/// constant frame-origin offset (e.g. from STFT centering) the wrapper
/// reports and the mapper applies; it is `0.0` for RMVPE's centered STFT.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelContract {
    pub sample_rate: u32,
    pub hop_samples: usize,
    pub origin_s: f64,
}

impl ModelContract {
    /// Seconds between native frames (`hop / rate`).
    pub fn frame_period_s(&self) -> f64 {
        self.hop_samples as f64 / self.sample_rate as f64
    }

    /// Native frame `k`'s time, in seconds.
    pub fn frame_time_s(&self, frame: usize) -> f64 {
        self.origin_s + frame as f64 * self.frame_period_s()
    }
}

/// RMVPE's contract: 16 kHz, 10 ms hop, centered STFT (`origin 0`).
pub const RMVPE_CONTRACT: ModelContract = ModelContract {
    sample_rate: 16_000,
    hop_samples: 160,
    origin_s: 0.0,
};

/// SwiftF0's contract (#59): 16 kHz, 256-sample (16 ms) hop, and a
/// `+127.5`-sample frame origin (`(1024 - 1) / 2 - 384`) the graph's
/// symmetric STFT padding imposes.
pub const SWIFTF0_CONTRACT: ModelContract = ModelContract {
    sample_rate: 16_000,
    hop_samples: 256,
    origin_s: 127.5 / 16_000.0,
};

/// A native-grid f0 contour: one value per native frame, `0.0` marking
/// unvoiced (#53). Non-finite and non-positive values also read unvoiced.
#[derive(Debug, Clone, PartialEq)]
pub struct NativeContour {
    pub contract: ModelContract,
    pub f0_hz: Vec<f64>,
}

impl NativeContour {
    /// The #47 interpolation rule at time `t`, in seconds: `t` is voiced iff
    /// its two bracketing native frames are both voiced; the value is linear
    /// in Hz between them. `t` before the first or after the last native
    /// frame has no bracket and is unvoiced.
    fn value_at(&self, t: f64) -> f64 {
        let values = &self.f0_hz;
        let hop_s = self.contract.frame_period_s();
        if values.len() < 2 || !(hop_s.is_finite() && hop_s > 0.0) || !t.is_finite() {
            return 0.0;
        }

        let position = ((t - self.contract.origin_s) / hop_s).floor();
        if !position.is_finite() || position < 0.0 {
            return 0.0;
        }
        let frame = position as usize;
        let (Some(&left), Some(&right)) = (values.get(frame), values.get(frame + 1)) else {
            return 0.0;
        };
        if !voiced(left) || !voiced(right) {
            return 0.0;
        }

        let left_time = self.contract.frame_time_s(frame);
        let weight = (t - left_time) / hop_s;
        left + (right - left) * weight
    }
}

/// An f0 track on the table grid; the same shape `kirafrqgen-core`'s
/// `F0Track` has, without depending on it.
#[derive(Debug, Clone, PartialEq)]
pub struct Track {
    pub frame_period_ms: f64,
    pub temporal_positions: Vec<f64>,
    pub f0_hz: Vec<f64>,
}

/// Map `native` onto the table grid of a `input_len_44100`-sample wav.
///
/// Always emits exactly [`table_frame_count`] frames; a table frame with no
/// two voiced bracketing native frames is `0.0`. The estimator does not
/// special-case the last frame: `build_table` forces it unvoiced and the key
/// is computed after, per #8.
pub fn map_to_table_grid(native: &NativeContour, input_len_44100: usize) -> Track {
    let frames = table_frame_count(input_len_44100);
    let period_s = TABLE_HOP_SAMPLES as f64 / TABLE_SAMPLE_RATE as f64;

    let f0_hz: Vec<f64> = (0..frames)
        .map(|frame| native.value_at(frame as f64 * period_s))
        .collect();

    Track {
        frame_period_ms: period_s * 1000.0,
        temporal_positions: (0..frames).map(|frame| frame as f64 * period_s).collect(),
        f0_hz,
    }
}

/// Unvoiced is `0.0`; any non-finite or non-positive value reads unvoiced.
fn voiced(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn contract_hop_160() -> ModelContract {
        RMVPE_CONTRACT
    }

    fn contour(values: Vec<f64>) -> NativeContour {
        NativeContour {
            contract: contract_hop_160(),
            f0_hz: values,
        }
    }

    fn approx(got: f64, want: f64) {
        assert!((got - want).abs() < 1e-9, "expected {want}, got {got}");
    }

    #[test]
    fn the_swiftf0_contract_is_16k_16ms_with_a_127_5_sample_origin() {
        assert_eq!(SWIFTF0_CONTRACT.sample_rate, 16_000);
        assert_eq!(SWIFTF0_CONTRACT.hop_samples, 256);
        approx(SWIFTF0_CONTRACT.frame_period_s(), 0.016);
        for frame in 0..4 {
            approx(
                SWIFTF0_CONTRACT.frame_time_s(frame),
                (127.5 + frame as f64 * 256.0) / 16_000.0,
            );
        }
    }

    #[test]
    fn frame_count_is_floor_len_over_256_plus_one() {
        for (length, frames) in [(0, 1), (1, 1), (255, 1), (256, 2), (1000, 4), (13_824, 55)] {
            assert_eq!(table_frame_count(length), frames, "L = {length}");
        }
    }

    #[test]
    fn the_track_sits_on_the_5_805_ms_grid() {
        let track = map_to_table_grid(&contour(vec![]), 1024);
        assert_eq!(track.f0_hz.len(), 5);
        approx(track.frame_period_ms, 256.0 / 44_100.0 * 1000.0);
        for (frame, &position) in track.temporal_positions.iter().enumerate() {
            approx(position, frame as f64 * 256.0 / 44_100.0);
        }
    }

    #[test]
    fn linear_hz_interpolation_between_native_frames() {
        // Native grid: frame k at k * 10 ms, f0 = 100 + 100k Hz. The table
        // frame at t = 0.004 s sits 40% from native frame 0 to frame 1.
        let track = map_to_table_grid(&contour(vec![100.0, 200.0, 300.0]), 2048);
        approx(track.f0_hz[0], 100.0);
        approx(track.f0_hz[1], 100.0 + 100.0 * (256.0 / 44_100.0 / 0.01));
        approx(
            track.f0_hz[2],
            200.0 + 100.0 * (512.0 / 44_100.0 / 0.01 - 1.0),
        );
    }

    #[test]
    fn a_single_voiced_native_frame_never_voices_a_table_frame() {
        let track = map_to_table_grid(&contour(vec![0.0, 220.0, 0.0]), 2048);
        assert!(
            track.f0_hz.iter().all(|&value| value == 0.0),
            "{:?}",
            track.f0_hz
        );
    }

    #[test]
    fn both_brackets_voiced_boundary_at_an_exact_native_frame() {
        // Native frames at 0/10/20 ms, all voiced: the bracket [1, 2] is the
        // last one, so table frames up to 20 ms interpolate and frames at or
        // past 20 ms have no bracket.
        let track = map_to_table_grid(&contour(vec![100.0, 200.0, 300.0]), 2048);
        let period = 256.0 / 44_100.0;
        for (frame, &value) in track.f0_hz.iter().enumerate() {
            let t = frame as f64 * period;
            if t >= 0.02 {
                assert_eq!(value, 0.0, "frame {frame} at t = {t}");
                continue;
            }
            let native = (t / 0.01).floor() as usize;
            let weight = t / 0.01 - native as f64;
            let left = 100.0 + 100.0 * native as f64;
            approx(value, left + 100.0 * weight);
        }
        // Table frame 2 sits at 11.61 ms, inside the voiced bracket [1, 2].
        approx(track.f0_hz[2], 200.0 + 100.0 * (2.0 * period / 0.01 - 1.0));
    }

    #[test]
    fn past_the_last_native_frame_is_unvoiced() {
        // Two native frames cover t < 10 ms only; with L = 1024 the later
        // table frames have no bracket and must be unvoiced.
        let track = map_to_table_grid(&contour(vec![220.0, 220.0]), 1024);
        assert_eq!(track.f0_hz.len(), 5);
        approx(track.f0_hz[0], 220.0);
        approx(track.f0_hz[1], 220.0);
        assert_eq!(&track.f0_hz[2..], &[0.0, 0.0, 0.0], "{:?}", track.f0_hz);
    }

    #[test]
    fn a_nonzero_origin_shifts_the_native_grid() {
        let native = NativeContour {
            contract: ModelContract {
                sample_rate: 16_000,
                hop_samples: 160,
                origin_s: 0.02,
            },
            f0_hz: vec![100.0, 200.0],
        };
        // 1024 samples = 23.2 ms of table grid: frames before 20 ms and at or
        // past 30 ms stay unvoiced; [20, 30 ms) interpolates.
        let track = map_to_table_grid(&native, 1024);
        let values = &track.f0_hz;
        for (frame, &value) in values.iter().enumerate() {
            let t = frame as f64 * 256.0 / 44_100.0;
            if (0.02..0.03).contains(&t) {
                approx(value, 100.0 + 100.0 * (t - 0.02) / 0.01);
            } else {
                assert_eq!(value, 0.0, "frame {frame} at t = {t}");
            }
        }
    }

    #[test]
    fn non_finite_and_negative_native_values_read_unvoiced() {
        let track = map_to_table_grid(&contour(vec![f64::NAN, f64::INFINITY, -5.0, 220.0]), 1024);
        assert!(
            track.f0_hz[..3].iter().all(|&value| value == 0.0),
            "{:?}",
            track.f0_hz
        );
    }

    #[test]
    fn an_empty_contour_is_all_unvoiced() {
        let track = map_to_table_grid(&contour(vec![]), 1000);
        assert_eq!(track.f0_hz, vec![0.0; 4]);
    }
}
