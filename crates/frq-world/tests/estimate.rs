use frq_world::{Estimator, F0Options, WorldError, estimate_f0, refine_f0_stonemask};

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

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() / expected < 0.01,
        "expected ~{expected} Hz, got {actual} Hz"
    );
}

#[test]
fn both_estimators_track_a_harmonic_tone_on_the_5ms_grid() {
    let samples = tone(220.0, 2.0);
    let options = F0Options::default();
    let expected_frames = (samples.len() as f64
        / (SAMPLE_RATE as f64 * options.frame_period_ms / 1000.0))
        .floor() as usize
        + 1;

    for estimator in [Estimator::Dio, Estimator::Harvest] {
        let mut track = estimate_f0(estimator, &samples, SAMPLE_RATE, &options).unwrap();
        assert_eq!(track.len(), expected_frames);
        assert!((track.temporal_positions[1] - track.temporal_positions[0] - 0.005).abs() < 1e-9);

        let mut voiced: Vec<f64> = track.voiced().collect();
        assert!(
            voiced.len() > track.len() / 2,
            "{estimator:?} found too few voiced frames"
        );
        assert_close(median(&mut voiced), 220.0);

        refine_f0_stonemask(&samples, SAMPLE_RATE, &mut track).unwrap();
        let mut refined: Vec<f64> = track.voiced().collect();
        assert_close(median(&mut refined), 220.0);
    }
}

#[test]
fn a_different_floor_moves_the_voiced_set() {
    let samples = tone(60.0, 2.0);
    let too_low_floor = estimate_f0(
        Estimator::Dio,
        &samples,
        SAMPLE_RATE,
        &F0Options {
            f0_floor_hz: 71.0,
            f0_ceiling_hz: 800.0,
            frame_period_ms: 5.0,
        },
    )
    .unwrap();
    let low_enough_floor = estimate_f0(
        Estimator::Dio,
        &samples,
        SAMPLE_RATE,
        &F0Options {
            f0_floor_hz: 50.0,
            f0_ceiling_hz: 800.0,
            frame_period_ms: 5.0,
        },
    )
    .unwrap();
    assert_eq!(too_low_floor.voiced().count(), 0);
    assert!(low_enough_floor.voiced().count() > 0);
}

#[test]
fn silence_is_unvoiced() {
    let samples = vec![0.0; SAMPLE_RATE as usize];
    for estimator in [Estimator::Dio, Estimator::Harvest] {
        let track = estimate_f0(estimator, &samples, SAMPLE_RATE, &F0Options::default()).unwrap();
        assert_eq!(track.voiced().count(), 0, "{estimator:?} voiced silence");
    }
}

#[test]
fn invalid_inputs_are_rejected() {
    let options = F0Options::default();
    assert_eq!(
        estimate_f0(Estimator::Dio, &[], SAMPLE_RATE, &options),
        Err(WorldError::EmptyInput)
    );
    assert_eq!(
        estimate_f0(Estimator::Dio, &[0.0; 64], 0, &options),
        Err(WorldError::InvalidSampleRate)
    );
    assert_eq!(
        estimate_f0(
            Estimator::Dio,
            &[0.0; 64],
            SAMPLE_RATE,
            &F0Options {
                f0_floor_hz: 800.0,
                f0_ceiling_hz: 71.0,
                frame_period_ms: 5.0,
            },
        ),
        Err(WorldError::InvalidOptions)
    );
}
