//! Contract for `d4c_aperiodicity0`, the raw D4C LoveTrain statistic behind
//! the tuned Harvest path's aperiodicity gate (#64).

use kirafrq_world_binding::{
    Estimator, F0Options, WorldError, d4c_aperiodicity0, estimate_f0, refine_f0_stonemask,
};

const SAMPLE_RATE: u32 = 44_100;

fn tone(frequency_hz: f64, seconds: f64) -> Vec<f64> {
    let count = (SAMPLE_RATE as f64 * seconds) as usize;
    (0..count)
        .map(|n| {
            let t = n as f64 / SAMPLE_RATE as f64;
            (1..=10)
                .map(|harmonic| {
                    (0.4 / harmonic as f64)
                        * (2.0 * std::f64::consts::PI * frequency_hz * harmonic as f64 * t).sin()
                })
                .sum()
        })
        .collect()
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

#[test]
fn the_statistic_is_high_where_a_harmonic_tone_is_voiced() {
    // A clean harmonic tone's energy sits well under 4 kHz, so the LoveTrain
    // ratio (100-4000 Hz vs 100-7900 Hz) stays near 1: periodic, voiced-like.
    let samples = tone(220.0, 1.0);
    let mut track = estimate_f0(
        Estimator::Harvest,
        &samples,
        SAMPLE_RATE,
        &F0Options::default(),
    )
    .unwrap();
    refine_f0_stonemask(&samples, SAMPLE_RATE, &mut track).unwrap();

    let statistic = d4c_aperiodicity0(
        &samples,
        SAMPLE_RATE,
        &track.temporal_positions,
        &track.f0_hz,
    )
    .unwrap();
    assert_eq!(statistic.len(), track.len(), "one value per track frame");

    let mut voiced: Vec<f64> = track
        .f0_hz
        .iter()
        .zip(&statistic)
        .filter_map(|(&f0_hz, &value)| (f0_hz > 0.0).then_some(value))
        .collect();
    assert!(!voiced.is_empty(), "the tone is voiced");
    assert!(
        median(&mut voiced) > 0.85,
        "D4C's own voiced split, got {}",
        median(&mut voiced)
    );
}

#[test]
fn unvoiced_frames_read_zero() {
    let samples = vec![0.0; SAMPLE_RATE as usize];
    let track = estimate_f0(
        Estimator::Harvest,
        &samples,
        SAMPLE_RATE,
        &F0Options::default(),
    )
    .unwrap();
    assert_eq!(track.voiced().count(), 0, "silence has no voiced frames");

    let statistic = d4c_aperiodicity0(
        &samples,
        SAMPLE_RATE,
        &track.temporal_positions,
        &track.f0_hz,
    )
    .unwrap();
    assert!(
        statistic.iter().all(|value| *value == 0.0),
        "frames with f0 == 0 read 0.0"
    );
}

#[test]
fn invalid_inputs_are_rejected() {
    assert_eq!(
        d4c_aperiodicity0(&[], SAMPLE_RATE, &[], &[]),
        Err(WorldError::EmptyInput)
    );

    let samples = tone(220.0, 0.1);
    assert_eq!(
        d4c_aperiodicity0(&samples, 0, &[], &[]),
        Err(WorldError::InvalidSampleRate)
    );
    assert_eq!(
        d4c_aperiodicity0(&samples, SAMPLE_RATE, &[], &[]),
        Err(WorldError::AnalysisFailed)
    );
    assert_eq!(
        d4c_aperiodicity0(&samples, SAMPLE_RATE, &[0.0], &[]),
        Err(WorldError::AnalysisFailed),
        "positions and f0 must be the same length"
    );
}
