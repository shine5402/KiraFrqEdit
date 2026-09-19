//! The SwiftF0 path over the bundled model (#69): session create/run, the raw
//! -audio adapter, the mandatory zero-pad, the #53 policy and the #47 mapping,
//! end to end. The real 398 KB model runs here, so the test is also the
//! provenance and availability guard for the shipped default ML tier.
//!
//! Synthetic smoke behavior mirrors #59's measurements: a harmonic stack
//! reads voiced at its fundamental with high confidence; digital silence
//! reads unvoiced.

#![cfg(feature = "ort")]

use kirafrq_ml_provider::grid::{
    SWIFTF0_CONTRACT, TABLE_HOP_SAMPLES, TABLE_SAMPLE_RATE, table_frame_count,
};
use kirafrq_ml_provider::model::ModelSource;
use kirafrq_ml_provider::policy::UvPolicy;
use kirafrq_ml_provider::swiftf0::{BUNDLED_MODEL, SwiftF0};

/// [`BUNDLED_MODEL`]'s SHA-256, pinned in #59.
const PINNED_SHA256: &str = "7e2390db8379cd9e1e2b22828e55b45b57c8559e4c8335678c717dc245c18176";

fn bundled() -> SwiftF0 {
    SwiftF0::new(
        ModelSource::Bundled {
            file_name: "swiftf0.onnx",
            bytes: BUNDLED_MODEL,
        },
        UvPolicy::swiftf0(None),
        1,
    )
}

/// A 10-harmonic stack at `hz`; #59 measured full voicing and high confidence
/// on this shape, level-independently.
fn harmonic_stack_16k(len: usize, hz: f64) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let t = index as f64 / 16_000.0;
            let mut value = 0.0;
            for harmonic in 1..=10 {
                value += (std::f64::consts::TAU * hz * harmonic as f64 * t).sin() / harmonic as f64;
            }
            (value * 0.1) as f32
        })
        .collect()
}

fn harmonic_stack_44100(len: usize, hz: f64) -> Vec<f64> {
    (0..len)
        .map(|index| {
            let t = index as f64 / 44_100.0;
            let mut value = 0.0;
            for harmonic in 1..=10 {
                value += (std::f64::consts::TAU * hz * harmonic as f64 * t).sin() / harmonic as f64;
            }
            value * 0.1
        })
        .collect()
}

#[test]
fn the_bundled_model_matches_the_pinned_commit() {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(BUNDLED_MODEL);
    let hex: String = digest.iter().map(|byte| format!("{byte:02x}")).collect();
    assert_eq!(
        hex, PINNED_SHA256,
        "the bundled swiftf0.onnx must be pinned"
    );
}

#[test]
fn session_creates_and_runs_the_bundled_model() {
    let estimator = bundled();
    estimator.ensure_session().unwrap();

    // 1 s of 16 kHz audio: exactly floor(16000 / 256) = 62 native frames.
    let native = estimator
        .native_estimate(&harmonic_stack_16k(16_000, 220.0))
        .unwrap();
    assert_eq!(native.len(), 16_000 / SWIFTF0_CONTRACT.hop_samples);
    assert!(
        native
            .iter()
            .all(|&value| value == 0.0 || (220.0 - 1.0..=220.0 + 1.0).contains(&value)),
        "a 220 Hz harmonic reads ~220 Hz, not noise: {:?}",
        &native[..native.len().min(8)]
    );
    let voiced = native.iter().filter(|&&value| value > 0.0).count();
    assert!(voiced > native.len() / 2, "mostly voiced: {voiced}");
}

#[test]
fn digital_silence_reads_unvoiced() {
    let native = bundled().native_estimate(&vec![0.0_f32; 16_000]).unwrap();
    assert!(
        native.iter().all(|&value| value == 0.0),
        "silence must never be voiced: {:?}",
        &native[..native.len().min(8)]
    );
}

#[test]
fn the_zero_pad_lets_a_sub_256_sample_input_run() {
    // A 200-sample 16 kHz clip is below the STFT kernel's minimum; the
    // adapter pads it to 256 and the graph emits one native frame instead of
    // failing (#59).
    let native = bundled().native_estimate(&vec![0.0_f32; 200]).unwrap();
    assert_eq!(native.len(), 1);
}

#[test]
fn a_harmonic_maps_onto_the_table_grid() {
    let samples = harmonic_stack_44100(44_100, 220.0);
    let track = bundled().estimate(&samples, None).unwrap();

    assert_eq!(track.f0_hz.len(), table_frame_count(samples.len()));
    let voiced: Vec<f64> = track
        .f0_hz
        .iter()
        .copied()
        .filter(|&value| value > 0.0)
        .collect();
    assert!(!voiced.is_empty(), "{:?}", track.f0_hz);
    let mean = voiced.iter().sum::<f64>() / voiced.len() as f64;
    assert!((mean - 220.0).abs() < 1.0, "mean voiced f0 {mean}");
    for &value in &voiced {
        assert!((71.0..=800.0).contains(&value), "out of range: {value}");
    }
}

#[test]
fn a_short_wav_short_circuits_without_running_the_model() {
    // A missing file would fail if the model were touched; the sub-hop
    // shortcut must never reach it.
    let missing = SwiftF0::new(
        ModelSource::File(std::path::PathBuf::from("does-not-exist.onnx")),
        UvPolicy::swiftf0(None),
        1,
    );
    let samples = harmonic_stack_44100(TABLE_HOP_SAMPLES - 1, 220.0);
    let track = missing.estimate(&samples, None).unwrap();
    assert_eq!(track.f0_hz, vec![0.0]);
}

#[test]
fn a_missing_model_reports_the_path() {
    let missing = SwiftF0::new(
        ModelSource::File(std::path::PathBuf::from("does-not-exist.onnx")),
        UvPolicy::swiftf0(None),
        1,
    );
    let samples = harmonic_stack_44100(TABLE_HOP_SAMPLES, 220.0);
    let error = missing.estimate(&samples, None).unwrap_err();
    let message = error.to_string();
    assert!(message.contains("does-not-exist.onnx"), "{message}");
}

#[test]
fn every_unvoiced_frame_is_exactly_zero_and_nothing_is_non_finite() {
    let samples = harmonic_stack_44100(44_100, 300.0);
    let track = bundled().estimate(&samples, None).unwrap();
    for &value in &track.f0_hz {
        assert!(value.is_finite(), "non-finite value: {value}");
        assert!(value >= 0.0, "negative value: {value}");
    }
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
    let samples = harmonic_stack_44100(44_100, 220.0);
    bundled()
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
fn the_resample_lands_on_the_table_grid_for_a_non_44_1_length() {
    let samples = harmonic_stack_44100(13_824, 220.0);
    let track = bundled().estimate(&samples, None).unwrap();
    assert_eq!(track.f0_hz.len(), 55);
    assert!((track.frame_period_ms - 256.0 / 44_100.0 * 1000.0).abs() < 1e-12);
    assert_eq!(TABLE_SAMPLE_RATE, 44_100);
}
