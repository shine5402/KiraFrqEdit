//! The RMVPE path over the committed tiny ONNX fixture (#57): session
//! create/run, the decoder, the #53 policy and the #47 mapping, end to end.
//! No real model runs in CI.
//!
//! The fixture's salience cycles with a period of 4 native frames: voiced at
//! ~441.5 Hz, voiced at ~220.4 Hz, out of range (>800 Hz) and below the
//! confidence threshold.

#![cfg(feature = "ort")]

use kirafrq_ml_provider::grid::{TABLE_HOP_SAMPLES, TABLE_SAMPLE_RATE, table_frame_count};
use kirafrq_ml_provider::policy::UvPolicy;
use kirafrq_ml_provider::rmvpe::Rmvpe;

fn fixture_path() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("rmvpe_tiny.onnx")
}

fn estimator() -> Rmvpe {
    Rmvpe::new(fixture_path(), UvPolicy::rmvpe(None), 1)
}

/// A 44.1 kHz sine long enough for `frames` table frames.
fn sine_44100(len: usize, hz: f64) -> Vec<f64> {
    (0..len)
        .map(|index| 0.5 * (std::f64::consts::TAU * hz * index as f64 / 44_100.0).sin())
        .collect()
}

#[test]
fn session_creates_and_runs_the_committed_fixture() {
    let samples = sine_44100(44_100, 440.0);
    let track = estimator().estimate(&samples, None).unwrap();
    assert_eq!(track.f0_hz.len(), table_frame_count(samples.len()));
}

#[test]
fn the_native_policy_reaches_the_mapped_track() {
    // The fixture's 16 kHz grid has one frame per 10 ms; the mapped track
    // must carry the voiced values through, and drop the out-of-range and
    // below-threshold native frames (never clamping).
    let samples = sine_44100(44_100, 440.0);
    let track = estimator().estimate(&samples, None).unwrap();

    let voiced: Vec<f64> = track
        .f0_hz
        .iter()
        .copied()
        .filter(|&value| value > 0.0)
        .collect();
    assert!(!voiced.is_empty(), "{:?}", track.f0_hz);
    for &value in &voiced {
        assert!(
            (71.0..=800.0).contains(&value),
            "out-of-range value in the mapped track: {value}"
        );
    }
    // The highest slot value the fixture can produce is ~441.5 Hz; a clamp
    // of the 891 Hz slot would show up above 441.6 Hz.
    let max = voiced.iter().copied().fold(f64::MIN, f64::max);
    assert!(max <= 442.0, "clamped out-of-range value: {max}");
}

#[test]
fn every_unvoiced_frame_is_exactly_zero_and_nothing_is_non_finite() {
    let samples = sine_44100(44_100, 300.0);
    let track = estimator().estimate(&samples, None).unwrap();
    for &value in &track.f0_hz {
        assert!(value.is_finite(), "non-finite value: {value}");
        assert!(value >= 0.0, "negative value: {value}");
    }
    assert!(track.f0_hz.contains(&0.0), "the fixture has unvoiced slots");
}

#[test]
fn a_short_wav_short_circuits_without_running_the_model() {
    // The model path would fail on a missing file; the sub-hop shortcut must
    // never touch it.
    let missing = Rmvpe::new(
        std::path::PathBuf::from("does-not-exist.onnx"),
        UvPolicy::rmvpe(None),
        1,
    );
    let samples = sine_44100(TABLE_HOP_SAMPLES - 1, 440.0);
    let track = missing.estimate(&samples, None).unwrap();
    assert_eq!(track.f0_hz, vec![0.0]);
}

#[test]
fn a_missing_model_reports_the_path() {
    let missing = Rmvpe::new(
        std::path::PathBuf::from("does-not-exist.onnx"),
        UvPolicy::rmvpe(None),
        1,
    );
    let samples = sine_44100(TABLE_HOP_SAMPLES, 440.0);
    let error = missing.estimate(&samples, None).unwrap_err();
    let message = error.to_string();
    assert!(message.contains("does-not-exist.onnx"), "{message}");
}

#[test]
fn the_resample_lands_on_the_table_grid_for_a_non_44_1_length() {
    // 13_824 samples = 54 table hops: the mapped track must have 55 frames
    // regardless of the intermediate 16 kHz length.
    let samples = sine_44100(13_824, 440.0);
    let track = estimator().estimate(&samples, None).unwrap();
    assert_eq!(track.f0_hz.len(), 55);
    assert!((track.frame_period_ms - 256.0 / 44_100.0 * 1000.0).abs() < 1e-12);
}

#[test]
fn a_progress_observer_sees_monotone_ticks_ending_at_total() {
    use kirafrq_ml_provider::ProgressObserver;
    use std::sync::Mutex;

    struct Watcher(Mutex<Vec<(usize, usize)>>);
    impl ProgressObserver for Watcher {
        fn report(&self, done: usize, total: usize) {
            self.0.lock().unwrap().push((done, total));
        }
    }

    let watcher = Watcher(Mutex::new(Vec::new()));
    let samples = sine_44100(44_100, 440.0);
    estimator()
        .estimate(&samples, Some(&watcher as &dyn ProgressObserver))
        .unwrap();

    let ticks = watcher.0.lock().unwrap().clone();
    assert!(!ticks.is_empty());
    assert!(
        ticks.windows(2).all(|pair| pair[0].0 <= pair[1].0),
        "{ticks:?}"
    );
    assert_eq!(ticks.last().copied(), Some((4, 4)), "{ticks:?}");
}

#[test]
fn the_table_sample_rate_matches_the_pipeline() {
    assert_eq!(TABLE_SAMPLE_RATE, 44_100);
}
