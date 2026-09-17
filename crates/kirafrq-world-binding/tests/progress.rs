use std::sync::Mutex;

use kirafrq_world_binding::{
    Estimator, F0Options, FrameObserver, ProgressStage, estimate_f0_with_observer,
    refine_f0_stonemask_with_observer,
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

#[derive(Default)]
struct Recording {
    events: Mutex<Vec<(ProgressStage, usize, usize)>>,
}

impl FrameObserver for Recording {
    fn report(&self, stage: ProgressStage, done: usize, total: usize) {
        self.events.lock().unwrap().push((stage, done, total));
    }
}

fn assert_monotonic_to_total(events: &[(ProgressStage, usize, usize)], track_len: usize) {
    assert!(!events.is_empty(), "the observer saw no events");
    let mut previous = 0;
    for (_, done, total) in events {
        assert!(*total > 0, "total must be positive");
        assert_eq!(*total, track_len, "every event names the pass total");
        assert!(
            *done > previous,
            "done moved backwards: {previous} then {done}"
        );
        assert!(*done <= *total, "done overshot the total");
        previous = *done;
    }
    assert_eq!(
        previous, track_len,
        "the final count must equal the track length"
    );
}

#[test]
fn progress_reports_every_frame_monotonically_for_dio_and_stonemask() {
    let samples = tone(220.0, 1.0);
    let options = F0Options::default();

    let estimate = Recording::default();
    let track = estimate_f0_with_observer(
        Estimator::Dio,
        &samples,
        SAMPLE_RATE,
        &options,
        Some(&estimate),
    )
    .unwrap();
    let estimate = estimate.events.lock().unwrap();
    assert!(
        estimate
            .iter()
            .all(|(stage, _, _)| *stage == ProgressStage::Estimate),
        "estimation reports the estimate stage"
    );
    assert_monotonic_to_total(&estimate, track.len());

    let mut track = track;
    let refine = Recording::default();
    refine_f0_stonemask_with_observer(&samples, SAMPLE_RATE, &mut track, Some(&refine)).unwrap();
    let refine = refine.events.lock().unwrap();
    assert!(
        refine
            .iter()
            .all(|(stage, _, _)| *stage == ProgressStage::Refine),
        "refinement reports the refine stage"
    );
    assert_monotonic_to_total(&refine, track.len());
}

#[test]
fn harvest_passes_report_monotonically_within_a_pass() {
    let samples = tone(220.0, 1.0);
    let options = F0Options::default();

    let recording = Recording::default();
    estimate_f0_with_observer(
        Estimator::Harvest,
        &samples,
        SAMPLE_RATE,
        &options,
        Some(&recording),
    )
    .unwrap();
    let events = recording.events.lock().unwrap();
    assert!(!events.is_empty(), "the observer saw no events");
    // One step per candidate pass: strictly increasing, ending at the pass
    // count, so the estimate phase composes without jumping back.
    let mut previous = 0;
    let mut total = 0;
    for (stage, done, pass_total) in events.iter() {
        assert_eq!(*stage, ProgressStage::Estimate);
        assert!(*pass_total > 0);
        total = *pass_total;
        assert!(
            *done > previous,
            "done moved backwards: {previous} then {done}"
        );
        assert!(*done <= *pass_total);
        previous = *done;
    }
    assert_eq!(previous, total, "the final count must equal the pass total");
}
