//! Contract for `kirafrqgen_core::generate`, `kirafrqgen_core::generate_wavs` and
//! `kirafrqgen_core::plan` over the settled policy set, exercised on real wavs in
//! a scratch folder with a fake [`F0Estimator`] (no WORLD).

use std::collections::BTreeSet;
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use hound::{SampleFormat, WavSpec, WavWriter};
use kirafrq_formats::{FrequencyTable, frq, mrq};
use kirafrqgen_core::{
    CancelToken, F0Estimator, F0Track, FileReport, GenerateOptions, GeneratorError, Progress,
    RunSummary, Sharing, Target, generate, generate_wavs, plan,
};

/// `8192 / 2^15` = 0.25, so the #8 amplitude of a constant wav is `8192.0`.
const SAMPLE: i16 = 8_192;

// --- scratch folder ---------------------------------------------------------

struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "kirafrqgen-core-{tag}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn join<P: AsRef<Path>>(&self, name: P) -> PathBuf {
        self.0.join(name)
    }

    fn sub(&self, name: &str) -> PathBuf {
        let path = self.0.join(name);
        fs::create_dir_all(&path).unwrap();
        path
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

// --- wav fixtures -----------------------------------------------------------

fn mono() -> WavSpec {
    WavSpec {
        channels: 1,
        sample_rate: 44_100,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    }
}

fn stereo_48k() -> WavSpec {
    WavSpec {
        channels: 2,
        sample_rate: 48_000,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    }
}

fn write_wav(scratch: &Scratch, name: &str, spec: WavSpec, samples: &[i16]) -> PathBuf {
    let path = scratch.join(name);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    let mut writer = WavWriter::create(&path, spec).unwrap();
    for sample in samples {
        writer.write_sample(*sample).unwrap();
    }
    writer.finalize().unwrap();
    path
}

/// A canonical generation table for fixture files written straight to disk.
fn fixture_table(frames: usize) -> FrequencyTable {
    FrequencyTable {
        sample_rate: 44_100,
        hop_samples: 256,
        f0_hz: (0..frames)
            .map(|index| if index % 2 == 0 { 110.0 } else { 0.0 })
            .collect(),
        amplitude: None,
        key_hz: 0.0,
    }
}

fn key(name: &str) -> Vec<u16> {
    name.encode_utf16().collect()
}

fn entry_nf0(entry: &mrq::Entry) -> i32 {
    i32::from_le_bytes(entry.raw()[0..4].try_into().unwrap())
}

fn entry_f0(entry: &mrq::Entry) -> Vec<f32> {
    (0..entry_nf0(entry) as usize)
        .map(|index| {
            f32::from_le_bytes(
                entry.raw()[12 + 4 * index..16 + 4 * index]
                    .try_into()
                    .unwrap(),
            )
        })
        .collect()
}

fn entry_i32(entry: &mrq::Entry, offset: usize) -> i32 {
    i32::from_le_bytes(entry.raw()[offset..offset + 4].try_into().unwrap())
}

// --- fake estimator ---------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
struct Call {
    length: usize,
    sample_rate: u32,
    frame_period_ms: f64,
}

struct FakeEstimator {
    pattern: Vec<f64>,
    calls: Mutex<Vec<Call>>,
    refined: AtomicUsize,
    refine_factor: f64,
    /// Sets the token when the n-th estimate (1-based) runs.
    cancel_on_call: Option<(usize, CancelToken)>,
    /// Whether this estimator advertises StoneMask support (#48).
    stone_mask: bool,
    /// Scripted raw D4C statistic per frame (#64); `None` = no statistic.
    aperiodicity0: Option<Vec<f64>>,
    /// Whether the statistic call fails (#64 per-file failure path).
    aperiodicity_error: bool,
    /// How many times the pipeline asked for the statistic (#64 wiring).
    aperiodicity_calls: AtomicUsize,
}

impl FakeEstimator {
    fn new(pattern: &[f64]) -> Self {
        Self {
            pattern: pattern.to_vec(),
            calls: Mutex::new(Vec::new()),
            refined: AtomicUsize::new(0),
            refine_factor: 1.0,
            cancel_on_call: None,
            stone_mask: true,
            aperiodicity0: None,
            aperiodicity_error: false,
            aperiodicity_calls: AtomicUsize::new(0),
        }
    }

    fn calls(&self) -> Vec<Call> {
        self.calls.lock().unwrap().clone()
    }

    fn refined_count(&self) -> usize {
        self.refined.load(Ordering::SeqCst)
    }

    fn aperiodicity_calls(&self) -> usize {
        self.aperiodicity_calls.load(Ordering::SeqCst)
    }
}

impl F0Estimator for FakeEstimator {
    fn estimate(
        &self,
        samples: &[f64],
        sample_rate: u32,
        frame_period_ms: f64,
        _observer: Option<&dyn kirafrq_world_binding::FrameObserver>,
    ) -> Result<F0Track, GeneratorError> {
        let call = self.calls.lock().unwrap().len() + 1;
        self.calls.lock().unwrap().push(Call {
            length: samples.len(),
            sample_rate,
            frame_period_ms,
        });
        if let Some((at, token)) = &self.cancel_on_call
            && call == *at
        {
            token.store(true, Ordering::SeqCst);
        }
        let frames = samples.len() / 256 + 1;
        Ok(F0Track {
            frame_period_ms,
            temporal_positions: (0..frames)
                .map(|index| index as f64 * frame_period_ms / 1000.0)
                .collect(),
            f0_hz: (0..frames)
                .map(|index| self.pattern[index % self.pattern.len()])
                .collect(),
        })
    }

    fn refine_stonemask(
        &self,
        _samples: &[f64],
        _sample_rate: u32,
        track: &mut F0Track,
        _observer: Option<&dyn kirafrq_world_binding::FrameObserver>,
    ) -> Result<(), GeneratorError> {
        self.refined.fetch_add(1, Ordering::SeqCst);
        for value in &mut track.f0_hz {
            if *value > 0.0 {
                *value *= self.refine_factor;
            }
        }
        Ok(())
    }

    fn supports_stonemask(&self) -> bool {
        self.stone_mask
    }

    fn aperiodicity0(
        &self,
        _samples: &[f64],
        _sample_rate: u32,
        track: &F0Track,
    ) -> Result<Option<Vec<f64>>, GeneratorError> {
        self.aperiodicity_calls.fetch_add(1, Ordering::SeqCst);
        if self.aperiodicity_error {
            return Err(GeneratorError::Estimation("D4C failed".to_string()));
        }
        Ok(self.aperiodicity0.as_ref().map(|values| {
            (0..track.f0_hz.len())
                .map(|index| values[index % values.len()])
                .collect()
        }))
    }
}

// --- run plumbing -----------------------------------------------------------

fn options(root: &Path, targets: &[Target]) -> GenerateOptions {
    GenerateOptions {
        root: root.to_path_buf(),
        targets: targets.iter().copied().collect(),
        overwrite: false,
        f0: kirafrqgen_core::F0Config::default(),
        jobs: 1,
        sharing: None,
        delete_llsm: true,
    }
}

#[derive(Default)]
struct Recording {
    started_files: Mutex<Vec<PathBuf>>,
    finished_files: Mutex<Vec<FileReport>>,
    finished_folders: Mutex<Vec<PathBuf>>,
    run_summary: Mutex<Option<RunSummary>>,
}

impl Progress for Recording {
    fn file_started(&self, wav: &Path) {
        self.started_files.lock().unwrap().push(wav.to_path_buf());
    }

    fn file_finished(&self, report: &FileReport) {
        self.finished_files.lock().unwrap().push(report.clone());
    }

    fn folder_finished(&self, folder: &Path) {
        self.finished_folders
            .lock()
            .unwrap()
            .push(folder.to_path_buf());
    }

    fn finished(&self, summary: &RunSummary) {
        *self.run_summary.lock().unwrap() = Some(summary.clone());
    }
}

fn run_result(
    opts: &GenerateOptions,
    estimator: &dyn F0Estimator,
) -> Result<RunSummary, GeneratorError> {
    let progress = Recording::default();
    let cancel: CancelToken = Arc::new(AtomicBool::new(false));
    generate(opts, estimator, &progress, &cancel)
}

fn run(opts: &GenerateOptions, estimator: &dyn F0Estimator) -> RunSummary {
    run_result(opts, estimator).unwrap()
}

// --- per-target policy values -----------------------------------------------

#[test]
fn writes_policy_correct_tables_for_all_targets() {
    let scratch = Scratch::new("all-targets");
    write_wav(&scratch, "A2.wav", mono(), &[SAMPLE; 1000]);
    let estimator = FakeEstimator::new(&[100.0, 200.0, 0.0, 400.0]);
    let opts = options(&scratch.0, &[Target::Frq, Target::Pmk, Target::Mrq]);

    let summary = run(&opts, &estimator);

    assert_eq!(summary.considered, 1);
    assert_eq!(summary.written, 1);
    assert_eq!(summary.skipped, 0);
    assert_eq!(summary.failed, []);
    assert_eq!(summary.warnings, []);
    assert!(!summary.cancelled);

    // WORLD is asked for the hop-derived frame period (#7) and StoneMasked.
    assert_eq!(estimator.calls().len(), 1);
    assert_eq!(estimator.calls()[0].sample_rate, 44_100);
    assert!((estimator.calls()[0].frame_period_ms - 5.804_988_662_131_519).abs() < 1e-12);
    assert_eq!(estimator.calls()[0].length, 1000);
    assert_eq!(estimator.refined_count(), 1);

    let table = frq::read(&scratch.join("A2_wav.frq")).unwrap();
    assert_eq!(
        table.f0_hz,
        [100.0, 200.0, 0.0, 0.0],
        "the trailing frame is forced unvoiced (#8)"
    );
    assert_eq!(table.key_hz, 150.0, "key = voiced mean (#8)");
    assert_eq!(
        table.amplitude,
        Some(vec![8192.0; 4]),
        "2^15 * mean(|x|) (#8)"
    );
    assert_eq!(table.hop_samples, 256);
    assert_eq!(table.sample_rate, 44_100);

    assert!(scratch.join("A2_wav.pmk").is_file());

    let desc = mrq::Desc::read(&mrq::desc_path(&scratch.0))
        .unwrap()
        .unwrap();
    assert_eq!(desc.len(), 1);
    let entry = &desc.entries()[0];
    assert_eq!(entry.name(), key("A2.wav"));
    assert_eq!(entry_nf0(entry), 3, "nf0 = frames - 1");
    assert_eq!(entry_f0(entry), [100.0, 200.0, 0.0]);
    assert_eq!(entry_i32(entry, 4), 44_100);
    assert_eq!(entry_i32(entry, 8), 256);
    let tail = 12 + 4 * 3;
    assert!(
        entry_i32(entry, tail) > 0,
        "timestamp = max(now, wav mtime) (#10)"
    );
    assert_eq!(entry_i32(entry, tail + 4), 0, "modified == 0");
}

#[test]
fn fill_missing_never_decodes_or_touches_existing_tables() {
    let scratch = Scratch::new("fill-missing");
    write_wav(&scratch, "A2.wav", mono(), &[SAMPLE; 512]);
    write_wav(&scratch, "A3.wav", mono(), &[SAMPLE; 512]);
    write_wav(&scratch, "A4.wav", mono(), &[SAMPLE; 512]);
    fs::write(scratch.join("A2_wav.frq"), b"canonical kept").unwrap();
    fs::write(scratch.join("A3.wav.frq"), b"alternate kept").unwrap();

    let estimator = FakeEstimator::new(&[220.0]);
    let summary = run(&options(&scratch.0, &[Target::Frq]), &estimator);

    assert_eq!(summary.considered, 3);
    assert_eq!(summary.written, 1);
    assert_eq!(summary.skipped, 2);
    assert_eq!(
        fs::read(scratch.join("A2_wav.frq")).unwrap(),
        b"canonical kept"
    );
    assert_eq!(
        fs::read(scratch.join("A3.wav.frq")).unwrap(),
        b"alternate kept"
    );
    assert!(scratch.join("A4_wav.frq").is_file());
    assert_eq!(estimator.calls().len(), 1, "only A4 was analyzed");
}

#[test]
fn overwrite_rewrites_the_canonical_and_deletes_the_alternate() {
    let scratch = Scratch::new("overwrite-alt");
    write_wav(&scratch, "A2.wav", mono(), &[SAMPLE; 1000]);
    fs::write(scratch.join("A2_wav.frq"), b"stale").unwrap();
    fs::write(scratch.join("A2.wav.frq"), b"stale alt").unwrap();

    let estimator = FakeEstimator::new(&[220.0]);
    let mut opts = options(&scratch.0, &[Target::Frq]);
    opts.overwrite = true;
    let summary = run(&opts, &estimator);

    assert_eq!(summary.written, 1);
    assert_eq!(summary.skipped, 0);
    assert!(!scratch.join("A2.wav.frq").exists(), "alt deleted");
    let table = frq::read(&scratch.join("A2_wav.frq")).unwrap();
    assert_eq!(table.f0_hz, [220.0, 220.0, 220.0, 0.0]);
}

#[test]
fn a_failed_write_never_deletes_the_alternate() {
    let scratch = Scratch::new("failed-write");
    write_wav(&scratch, "A2.wav", mono(), &[SAMPLE; 512]);
    // The canonical path is a directory, so the write fails after the
    // existence check passed.
    fs::create_dir(scratch.join("A2_wav.frq")).unwrap();
    fs::write(scratch.join("A2.wav.frq"), b"alt").unwrap();

    let estimator = FakeEstimator::new(&[220.0]);
    let mut opts = options(&scratch.0, &[Target::Frq]);
    opts.overwrite = true;
    let summary = run(&opts, &estimator);

    assert_eq!(summary.written, 0);
    assert_eq!(summary.failed.len(), 1, "{:?}", summary.failed);
    assert!(summary.failed[0].1.contains("A2_wav.frq"));
    assert_eq!(fs::read(scratch.join("A2.wav.frq")).unwrap(), b"alt");
}

#[test]
fn pmk_is_canonical_only() {
    let scratch = Scratch::new("pmk-canonical");
    write_wav(&scratch, "A2.wav", mono(), &[SAMPLE; 512]);
    fs::write(scratch.join("A2.pmk"), b"not a spelling we read").unwrap();

    let estimator = FakeEstimator::new(&[220.0]);
    let summary = run(&options(&scratch.0, &[Target::Pmk]), &estimator);

    assert_eq!(summary.written, 1);
    assert!(scratch.join("A2_wav.pmk").is_file());
    assert_eq!(
        fs::read(scratch.join("A2.pmk")).unwrap(),
        b"not a spelling we read"
    );
}

// --- mrq merge --------------------------------------------------------------

#[test]
fn mrq_merge_preserves_foreign_entries_and_appends() {
    let scratch = Scratch::new("merge");
    write_wav(&scratch, "A2.wav", mono(), &[SAMPLE; 1000]);

    let mut fixture = mrq::Desc::empty();
    let foreign = mrq::Entry::from_table(&key("foreign.wav"), &fixture_table(10), 100).unwrap();
    let tombstone = mrq::Entry::from_table(&[], &fixture_table(6), 100).unwrap();
    fixture.upsert(foreign.clone());
    fixture.upsert(tombstone.clone());
    fixture.write(&mrq::desc_path(&scratch.0)).unwrap();

    let estimator = FakeEstimator::new(&[220.0]);
    let summary = run(&options(&scratch.0, &[Target::Mrq]), &estimator);
    assert_eq!(summary.written, 1);

    let desc = mrq::Desc::read(&mrq::desc_path(&scratch.0))
        .unwrap()
        .unwrap();
    assert_eq!(desc.len(), 3, "entry_count never decrements");
    assert_eq!(desc.entries()[0], foreign, "foreign entry byte-identical");
    assert_eq!(desc.entries()[1], tombstone, "tombstone preserved");
    assert_eq!(desc.entries()[2].name(), key("A2.wav"), "appended");
}

#[test]
fn one_sided_sharing_entries_count_as_absent_and_updates_both() {
    // CP936 mojibake of a Japanese filename, the state a shared bank has.
    const JAPANESE: &str = "_\u{3042}\u{304b}\u{3055}\u{305f}\u{306a}.wav";
    const LOCAL: &str = "_\u{5041}\u{5050}\u{505d}\u{5068}\u{5074}.wav";
    let sharing = Sharing::new(936);
    let keys = mrq::entry_keys(OsStr::new(LOCAL), Some(&sharing));
    assert_eq!(keys, [key(LOCAL), key(JAPANESE)]);

    let scratch = Scratch::new("sharing-onesided");
    write_wav(&scratch, LOCAL, mono(), &[SAMPLE; 1000]);

    let mut fixture = mrq::Desc::empty();
    fixture.upsert(mrq::Entry::from_table(&keys[0], &fixture_table(10), 100).unwrap());
    fixture.write(&mrq::desc_path(&scratch.0)).unwrap();

    let estimator = FakeEstimator::new(&[220.0]);
    let mut opts = options(&scratch.0, &[Target::Mrq]);
    opts.sharing = Some(sharing);

    let summary = run(&opts, &estimator);
    assert_eq!(summary.written, 1, "one-sided counts as absent (#10)");
    let desc = mrq::Desc::read(&mrq::desc_path(&scratch.0))
        .unwrap()
        .unwrap();
    assert!(keys.iter().all(|key| desc.has(key)), "both names exist");
    assert_eq!(desc.len(), 2);

    let summary = run(&opts, &estimator);
    assert_eq!(summary.skipped, 1, "both names exist: fill-missing skips");
    assert_eq!(summary.written, 0);
    assert_eq!(estimator.calls().len(), 1, "and no analysis runs");

    opts.overwrite = true;
    let summary = run(&opts, &estimator);
    assert_eq!(summary.written, 1);
    let desc = mrq::Desc::read(&mrq::desc_path(&scratch.0))
        .unwrap()
        .unwrap();
    assert_eq!(desc.len(), 2, "no duplicates on regeneration");
    assert_eq!(
        desc.entries()[0].raw(),
        desc.entries()[1].raw(),
        "both names carry the same f0"
    );
}

// --- llsm -------------------------------------------------------------------

#[test]
fn llsm_is_deleted_only_for_mrq_writes() {
    let scratch = Scratch::new("llsm");

    let write = scratch.sub("write");
    write_wav(&scratch, "write/A2.wav", mono(), &[SAMPLE; 1000]);

    let existing = scratch.sub("existing");
    write_wav(&scratch, "existing/A3.wav", mono(), &[SAMPLE; 1000]);
    fs::write(
        scratch.join("existing/A3_wav.frq"),
        frq::to_bytes(&fixture_table(4)),
    )
    .unwrap();

    let pmk = scratch.sub("pmk");
    write_wav(&scratch, "pmk/A4.wav", mono(), &[SAMPLE; 1000]);

    let mrq_folder = scratch.sub("mrq");
    write_wav(&scratch, "mrq/A5.wav", mono(), &[SAMPLE; 1000]);

    for folder in ["write", "existing", "pmk", "mrq"] {
        fs::write(scratch.join(format!("{folder}/A2.wav.llsm")), b"x").unwrap();
        fs::write(scratch.join(format!("{folder}/A3.wav.llsm")), b"x").unwrap();
        fs::write(scratch.join(format!("{folder}/A4.wav.llsm")), b"x").unwrap();
        fs::write(scratch.join(format!("{folder}/A5.wav.llsm")), b"x").unwrap();
    }
    let cache = |folder: &str, name: &str| scratch.join(format!("{folder}/{name}.wav.llsm"));

    let estimator = FakeEstimator::new(&[220.0]);
    let summary = run(&options(&write, &[Target::Frq]), &estimator);
    assert_eq!(summary.written, 1);
    assert!(
        cache("write", "A2").exists(),
        "frq is not moresampler's f0 source, so a write leaves the cache alone"
    );

    let summary = run(&options(&existing, &[Target::Frq]), &estimator);
    assert_eq!(summary.skipped, 1);
    assert!(cache("existing", "A3").exists(), "no write, no delete");

    let summary = run(&options(&pmk, &[Target::Pmk]), &estimator);
    assert_eq!(summary.written, 1);
    assert!(cache("pmk", "A4").exists(), "pmk never invalidates f0");

    let summary = run(&options(&mrq_folder, &[Target::Mrq]), &estimator);
    assert_eq!(summary.written, 1);
    assert!(!cache("mrq", "A5").exists(), "mrq write deletes");
}

#[test]
fn llsm_deletion_can_be_disabled() {
    let scratch = Scratch::new("llsm-off");
    write_wav(&scratch, "A2.wav", mono(), &[SAMPLE; 1000]);
    fs::write(scratch.join("A2.wav.llsm"), b"x").unwrap();

    let estimator = FakeEstimator::new(&[220.0]);
    let mut opts = options(&scratch.0, &[Target::Mrq]);
    opts.delete_llsm = false;
    let summary = run(&opts, &estimator);

    assert_eq!(summary.written, 1);
    assert!(scratch.join("A2.wav.llsm").exists());
    assert_eq!(summary.warnings, []);
}

// --- cancellation -----------------------------------------------------------

#[test]
fn cancellation_before_decode_writes_nothing() {
    let scratch = Scratch::new("cancel-early");
    write_wav(&scratch, "A2.wav", mono(), &[SAMPLE; 1000]);
    write_wav(&scratch, "A3.wav", mono(), &[SAMPLE; 1000]);

    let estimator = FakeEstimator::new(&[220.0]);
    let opts = options(&scratch.0, &[Target::Frq]);
    let progress = Recording::default();
    let cancel: CancelToken = Arc::new(AtomicBool::new(true));
    let summary = generate(&opts, &estimator, &progress, &cancel).unwrap();

    assert!(summary.cancelled);
    assert_eq!(summary.considered, 2);
    assert_eq!(summary.written, 0);
    assert_eq!(summary.skipped, 0);
    assert!(estimator.calls().is_empty(), "no decode or analysis");
    assert!(!scratch.join("A2_wav.frq").exists());
    assert!(
        progress
            .run_summary
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .cancelled
    );
}

#[test]
fn cancellation_before_writes_stops_the_file_after_analysis() {
    let scratch = Scratch::new("cancel-write");
    write_wav(&scratch, "A2.wav", mono(), &[SAMPLE; 1000]);
    write_wav(&scratch, "A3.wav", mono(), &[SAMPLE; 1000]);

    let cancel: CancelToken = Arc::new(AtomicBool::new(false));
    let mut estimator = FakeEstimator::new(&[220.0]);
    estimator.cancel_on_call = Some((1, Arc::clone(&cancel)));
    let opts = options(&scratch.0, &[Target::Frq]);
    let progress = Recording::default();
    let summary = generate(&opts, &estimator, &progress, &cancel).unwrap();

    assert!(summary.cancelled);
    assert_eq!(summary.written, 0);
    assert_eq!(estimator.calls().len(), 1, "the first wav was analyzed");
    assert!(!scratch.join("A2_wav.frq").exists(), "but never written");
    assert!(!scratch.join("A3_wav.frq").exists());
}

#[test]
fn cancellation_before_the_mrq_merge_drops_the_contributions() {
    let scratch = Scratch::new("cancel-merge");
    write_wav(&scratch, "A2.wav", mono(), &[SAMPLE; 1000]);
    write_wav(&scratch, "A3.wav", mono(), &[SAMPLE; 1000]);

    let cancel: CancelToken = Arc::new(AtomicBool::new(false));
    let mut estimator = FakeEstimator::new(&[220.0]);
    // A2 builds its mrq contribution; A3's estimate then sets the token, so
    // phase A ends with A2's report waiting on a merge that must not run.
    estimator.cancel_on_call = Some((2, Arc::clone(&cancel)));
    let opts = options(&scratch.0, &[Target::Mrq]);
    let progress = Recording::default();
    let summary = generate(&opts, &estimator, &progress, &cancel).unwrap();

    assert!(summary.cancelled);
    assert_eq!(summary.written, 0, "the merge never ran");
    assert!(!mrq::desc_path(&scratch.0).exists());
    assert_eq!(estimator.calls().len(), 2);
    let finished = progress.finished_files.lock().unwrap();
    assert_eq!(finished.len(), 2, "both reports fired");
    assert!(finished.iter().all(|report| report.cancelled));
}

// --- warnings and failures --------------------------------------------------

#[test]
fn decode_warnings_and_failures_are_per_file_and_never_abort() {
    let scratch = Scratch::new("warn-fail");
    write_wav(&scratch, "good.wav", mono(), &[SAMPLE; 1000]);
    write_wav(&scratch, "wide.wav", stereo_48k(), &[SAMPLE; 2000]);
    fs::write(scratch.join("broken.wav"), b"not a wav").unwrap();

    let estimator = FakeEstimator::new(&[220.0]);
    let summary = run(&options(&scratch.0, &[Target::Frq]), &estimator);

    assert_eq!(summary.considered, 3);
    assert_eq!(summary.written, 2);
    assert_eq!(summary.failed.len(), 1);
    assert!(summary.failed[0].0.ends_with("broken.wav"));
    assert!(!summary.failed[0].1.is_empty());

    assert_eq!(summary.warnings.len(), 2, "{:?}", summary.warnings);
    assert!(
        summary
            .warnings
            .iter()
            .all(|(path, _)| path.ends_with("wide.wav"))
    );
    assert!(summary.warnings[0].1.contains("resampled"));
    assert!(summary.warnings[1].1.contains("channels"));
    assert!(scratch.join("wide_wav.frq").is_file());
}

// --- degenerate wavs --------------------------------------------------------

#[test]
fn sub_hop_wavs_write_degenerate_frq_and_get_no_mrq_entry() {
    let scratch = Scratch::new("sub-hop");
    write_wav(&scratch, "tiny.wav", mono(), &[SAMPLE; 100]);

    let estimator = FakeEstimator::new(&[220.0]);
    let opts = options(&scratch.0, &[Target::Frq, Target::Mrq]);
    let progress = Recording::default();
    let cancel: CancelToken = Arc::new(AtomicBool::new(false));
    let summary = generate(&opts, &estimator, &progress, &cancel).unwrap();

    assert_eq!(summary.written, 1, "frq still gets its degenerate table");
    let table = frq::read(&scratch.join("tiny_wav.frq")).unwrap();
    assert_eq!(table.f0_hz, [0.0], "the single frame is forced unvoiced");
    assert_eq!(table.key_hz, 0.0);
    assert_eq!(table.amplitude, Some(vec![8192.0]));
    assert!(
        !mrq::desc_path(&scratch.0).exists(),
        "no mrq entry, no merge-write"
    );

    let finished = progress.finished_files.lock().unwrap();
    let report = &finished[0];
    assert!(report.written.contains(&Target::Frq));
    assert!(
        report.no_entry.contains(&Target::Mrq),
        "a sub-hop wav is not 'existing': it can never carry an mrq table"
    );
    assert!(report.existing.is_empty());
}

#[test]
fn zero_length_wavs_are_skipped_without_touching_the_estimator() {
    let scratch = Scratch::new("zero");
    let path = scratch.join("empty.wav");
    let writer = WavWriter::create(&path, mono()).unwrap();
    writer.finalize().unwrap();

    let estimator = FakeEstimator::new(&[220.0]);
    let summary = run(&options(&scratch.0, &[Target::Frq]), &estimator);

    assert_eq!(summary.considered, 1);
    assert_eq!(summary.skipped, 1);
    assert_eq!(summary.written, 0);
    assert!(estimator.calls().is_empty());
    assert!(!scratch.join("empty_wav.frq").exists());
}

// --- corrupt desc.mrq -------------------------------------------------------

#[test]
fn corrupt_desc_is_renamed_aside_and_replaced() {
    let scratch = Scratch::new("corrupt-desc");
    write_wav(&scratch, "A2.wav", mono(), &[SAMPLE; 1000]);
    write_wav(&scratch, "A3.wav", mono(), &[SAMPLE; 1000]);
    fs::write(mrq::desc_path(&scratch.0), b"nope, not mrq").unwrap();

    let estimator = FakeEstimator::new(&[220.0]);
    let summary = run(&options(&scratch.0, &[Target::Mrq]), &estimator);

    assert_eq!(summary.written, 2);
    assert_eq!(
        summary.warnings.len(),
        1,
        "one folder-level warning per corrupt desc.mrq, not one per wav: {:?}",
        summary.warnings
    );
    assert!(summary.warnings[0].1.contains("corrupt"));
    assert!(summary.warnings[0].1.contains("renamed to"));

    let backup = fs::read_dir(&scratch.0)
        .unwrap()
        .filter_map(Result::ok)
        .find(|entry| {
            entry
                .file_name()
                .to_string_lossy()
                .starts_with("desc.mrq.corrupt-")
        })
        .expect("backup exists");
    assert_eq!(fs::read(backup.path()).unwrap(), b"nope, not mrq");

    let desc = mrq::Desc::read(&mrq::desc_path(&scratch.0))
        .unwrap()
        .unwrap();
    assert!(desc.has(&key("A2.wav")));
    assert!(desc.has(&key("A3.wav")));
}

// --- dry run ----------------------------------------------------------------

#[test]
fn plan_reports_existing_tables_without_decoding_or_writing() {
    let scratch = Scratch::new("plan");
    write_wav(&scratch, "A2.wav", mono(), &[SAMPLE; 1000]);
    write_wav(&scratch, "A3.wav", mono(), &[SAMPLE; 1000]);
    write_wav(&scratch, "A4.wav", mono(), &[SAMPLE; 1000]);
    fs::write(scratch.join("A2_wav.frq"), b"x").unwrap();
    fs::write(scratch.join("A2_wav.pmk"), b"x").unwrap();
    fs::write(scratch.join("A3.wav.frq"), b"x").unwrap();

    let opts = options(&scratch.0, &[Target::Frq, Target::Pmk, Target::Mrq]);
    let plan = plan(&opts).unwrap();

    assert_eq!(plan.files.len(), 3, "sorted scan order");
    let by_name = |name: &str| {
        plan.files
            .iter()
            .find(|file| file.wav.ends_with(name))
            .unwrap()
    };
    assert_eq!(
        by_name("A2.wav").existing,
        [Target::Frq, Target::Pmk].into()
    );
    assert_eq!(by_name("A3.wav").existing, [Target::Frq].into());
    assert_eq!(by_name("A4.wav").existing, BTreeSet::new());
    assert!(!mrq::desc_path(&scratch.0).exists(), "plan writes nothing");
    assert!(plan.files.iter().all(|file| file.warnings.is_empty()));
}

#[test]
fn plan_notes_a_corrupt_desc_as_absent() {
    let scratch = Scratch::new("plan-corrupt");
    write_wav(&scratch, "A2.wav", mono(), &[SAMPLE; 512]);
    write_wav(&scratch, "A3.wav", mono(), &[SAMPLE; 512]);
    fs::write(mrq::desc_path(&scratch.0), b"nope").unwrap();

    let plan = plan(&options(&scratch.0, &[Target::Mrq])).unwrap();

    assert!(plan.files[0].existing.is_empty());
    assert_eq!(
        plan.files[0].warnings.len(),
        1,
        "the corrupt-desc note rides the folder's first wav"
    );
    assert!(plan.files[0].warnings[0].contains("corrupt"));
    assert!(plan.files[1].warnings.is_empty());
    assert_eq!(
        fs::read(mrq::desc_path(&scratch.0)).unwrap(),
        b"nope",
        "plan never touches the file"
    );
}

// --- roots and configuration ------------------------------------------------

#[test]
fn a_single_wav_root_is_a_one_file_run() {
    let scratch = Scratch::new("single-root");
    let wav = write_wav(&scratch, "A2.wav", mono(), &[SAMPLE; 512]);

    let summary = run(
        &options(&wav, &[Target::Frq]),
        &FakeEstimator::new(&[220.0]),
    );

    assert_eq!(summary.considered, 1);
    assert_eq!(summary.written, 1);
    assert!(scratch.join("A2_wav.frq").is_file());
}

// --- explicit file lists -----------------------------------------------------

#[test]
fn generate_wavs_processes_only_the_listed_wavs() {
    let scratch = Scratch::new("list-subset");
    let a2 = write_wav(&scratch, "A2.wav", mono(), &[SAMPLE; 512]);
    write_wav(&scratch, "A3.wav", mono(), &[SAMPLE; 512]);
    let estimator = FakeEstimator::new(&[220.0]);
    let progress = Recording::default();
    let cancel: CancelToken = Arc::new(AtomicBool::new(false));

    let summary = generate_wavs(
        &options(&scratch.0, &[Target::Frq]),
        std::slice::from_ref(&a2),
        &estimator,
        &progress,
        &cancel,
    )
    .unwrap();

    assert_eq!(summary.considered, 1);
    assert_eq!(summary.written, 1);
    assert_eq!(
        estimator.calls().len(),
        1,
        "only the listed wav is analyzed"
    );
    assert_eq!(progress.started_files.lock().unwrap().as_slice(), [a2]);
    assert!(scratch.join("A2_wav.frq").is_file());
    assert!(!scratch.join("A3_wav.frq").exists());
}

#[test]
fn generate_wavs_merges_mrq_for_the_selected_subset_only() {
    let scratch = Scratch::new("list-mrq");
    let a2 = write_wav(&scratch, "A2.wav", mono(), &[SAMPLE; 512]);
    let a3 = write_wav(&scratch, "A3.wav", mono(), &[SAMPLE; 512]);
    write_wav(&scratch, "A4.wav", mono(), &[SAMPLE; 512]);
    let estimator = FakeEstimator::new(&[220.0]);
    let progress = Recording::default();
    let cancel: CancelToken = Arc::new(AtomicBool::new(false));

    let summary = generate_wavs(
        &options(&scratch.0, &[Target::Mrq]),
        &[a2, a3],
        &estimator,
        &progress,
        &cancel,
    )
    .unwrap();

    assert_eq!(summary.considered, 2);
    assert_eq!(summary.written, 2);
    let desc = mrq::Desc::read(&mrq::desc_path(&scratch.0))
        .unwrap()
        .unwrap();
    assert_eq!(desc.len(), 2);
    assert!(desc.has(&key("A2.wav")));
    assert!(desc.has(&key("A3.wav")));
    assert!(!desc.has(&key("A4.wav")));
}

#[test]
fn generate_wavs_rejects_an_empty_selection() {
    let scratch = Scratch::new("list-empty");
    let estimator = FakeEstimator::new(&[220.0]);
    let progress = Recording::default();
    let cancel: CancelToken = Arc::new(AtomicBool::new(false));

    let result = generate_wavs(
        &options(&scratch.0, &[Target::Frq]),
        &[],
        &estimator,
        &progress,
        &cancel,
    );

    assert!(matches!(result, Err(GeneratorError::Config(_))));
}

#[test]
fn missing_and_wavless_roots_are_fatal() {
    let scratch = Scratch::new("roots");
    let estimator = FakeEstimator::new(&[220.0]);

    let missing = scratch.join("missing");
    assert!(matches!(
        run_result(&options(&missing, &[Target::Frq]), &estimator),
        Err(GeneratorError::Scan { .. })
    ));
    assert!(
        matches!(
            run_result(&options(&missing, &[]), &estimator),
            Err(GeneratorError::Config(_))
        ),
        "config is validated before the scan"
    );

    let empty = scratch.sub("empty");
    assert!(matches!(
        run_result(&options(&empty, &[Target::Frq]), &estimator),
        Err(GeneratorError::NoWavs { .. })
    ));

    write_wav(&scratch, "A2.wav", mono(), &[SAMPLE; 512]);
    assert!(matches!(
        run_result(&options(&scratch.0, &[]), &estimator),
        Err(GeneratorError::Config(_))
    ));
}

#[test]
fn an_invalid_energy_gate_ratio_is_a_config_error() {
    let scratch = Scratch::new("gate-ratio");
    write_wav(&scratch, "A2.wav", mono(), &[SAMPLE; 512]);
    let estimator = FakeEstimator::new(&[220.0]);

    for ratio in [f64::NAN, -0.01] {
        let mut opts = options(&scratch.0, &[Target::Frq]);
        opts.f0.energy_gate_ratio = ratio;
        assert!(
            matches!(
                run_result(&opts, &estimator),
                Err(GeneratorError::Config(_))
            ),
            "ratio {ratio} must be rejected"
        );
    }
}

// --- estimator wiring -------------------------------------------------------

#[test]
fn stone_mask_refinement_is_wired_to_the_config() {
    let scratch = Scratch::new("stonemask");
    write_wav(&scratch, "A2.wav", mono(), &[SAMPLE; 1000]);

    let mut estimator = FakeEstimator::new(&[100.0, 200.0, 0.0, 400.0]);
    estimator.refine_factor = 2.0;

    let opts = options(&scratch.0, &[Target::Frq]);
    assert!(opts.f0.stone_mask, "the default has StoneMask on");
    run(&opts, &estimator);
    assert_eq!(estimator.refined_count(), 1);
    let table = frq::read(&scratch.join("A2_wav.frq")).unwrap();
    assert_eq!(table.f0_hz, [200.0, 400.0, 0.0, 0.0]);
    assert_eq!(table.key_hz, 300.0);

    let mut off = options(&scratch.0, &[Target::Frq]);
    off.f0.stone_mask = false;
    off.overwrite = true;
    run(&off, &estimator);
    assert_eq!(estimator.refined_count(), 1, "no refine with StoneMask off");
    let table = frq::read(&scratch.join("A2_wav.frq")).unwrap();
    assert_eq!(table.f0_hz, [100.0, 200.0, 0.0, 0.0]);
}

#[test]
fn an_estimator_without_stonemask_is_never_refined_even_with_the_flag_on() {
    let scratch = Scratch::new("stonemask-capability");
    write_wav(&scratch, "A2.wav", mono(), &[SAMPLE; 1000]);

    let mut estimator = FakeEstimator::new(&[100.0, 200.0, 0.0, 400.0]);
    estimator.refine_factor = 2.0;
    estimator.stone_mask = false;

    let opts = options(&scratch.0, &[Target::Frq]);
    assert!(opts.f0.stone_mask, "the flag is on; the capability decides");
    run(&opts, &estimator);

    assert_eq!(
        estimator.refined_count(),
        0,
        "an estimator without StoneMask never refines (#48)"
    );
    let table = frq::read(&scratch.join("A2_wav.frq")).unwrap();
    assert_eq!(table.f0_hz, [100.0, 200.0, 0.0, 0.0]);
}

#[test]
fn the_energy_gate_is_world_only() {
    // #53: the WORLD energy gate is a WORLD workaround; an ML estimator's
    // voicing policy must survive `world_quirks` untouched.
    let scratch = Scratch::new("ml-no-energy-gate");
    write_wav(&scratch, "A2.wav", mono(), &gated_samples());
    let estimator = FakeEstimator::new(&[110.0, 220.0, 330.0, 440.0, 550.0]);

    let mut opts = options(&scratch.0, &[Target::Frq]);
    opts.f0.estimator = kirafrqgen_core::Estimator::Rmvpe;
    assert!(opts.f0.world_quirks, "the default is the tuned path");
    run(&opts, &estimator);

    let table = frq::read(&scratch.join("A2_wav.frq")).unwrap();
    assert_eq!(
        table.f0_hz,
        [110.0, 220.0, 330.0, 440.0, 0.0],
        "no energy gate on the ML path"
    );
}

// --- ML estimator factory (#48/#49) -----------------------------------------

#[test]
fn the_factory_builds_the_world_pair() {
    let config = kirafrqgen_core::F0Config {
        estimator: kirafrqgen_core::Estimator::Dio,
        ..kirafrqgen_core::F0Config::default()
    };
    let estimator = kirafrqgen_core::build_estimator(&config, 1).unwrap();
    assert!(estimator.supports_stonemask(), "WORLD refines");

    let config = kirafrqgen_core::F0Config {
        estimator: kirafrqgen_core::Estimator::Harvest,
        ..kirafrqgen_core::F0Config::default()
    };
    let estimator = kirafrqgen_core::build_estimator(&config, 1).unwrap();
    assert!(estimator.supports_stonemask(), "WORLD refines");
}

#[test]
fn selecting_rmvpe_without_a_model_is_an_early_config_error() {
    let config = kirafrqgen_core::F0Config {
        estimator: kirafrqgen_core::Estimator::Rmvpe,
        ml: kirafrqgen_core::MlConfig {
            model_path: Some(PathBuf::from("definitely-not-a-model.onnx")),
            confidence_threshold: None,
        },
        ..kirafrqgen_core::F0Config::default()
    };
    match kirafrqgen_core::build_estimator(&config, 1) {
        Err(GeneratorError::Config(message)) => {
            // With `ml` on the error names the path; the compat build reports
            // the missing feature. Both are early config errors (#48).
            if kirafrqgen_core::ML_SUPPORTED {
                assert!(message.contains("definitely-not-a-model.onnx"), "{message}");
            } else {
                assert!(message.contains("no ML estimator support"), "{message}");
            }
        }
        Err(other) => panic!("expected a config error, got {other}"),
        Ok(_) => panic!("expected a config error, got an estimator"),
    }
}

#[test]
fn the_ml_capability_predicates_are_consistent() {
    // `ml_available` must never claim availability in a build without the
    // feature, and a model path is existence-checked.
    let config = kirafrqgen_core::F0Config::default();
    if !kirafrqgen_core::ML_SUPPORTED {
        assert!(!kirafrqgen_core::ml_available(&config));
    }
    let missing = kirafrqgen_core::F0Config {
        ml: kirafrqgen_core::MlConfig {
            model_path: Some(PathBuf::from("no-such-model.onnx")),
            confidence_threshold: None,
        },
        ..kirafrqgen_core::F0Config::default()
    };
    assert!(!kirafrqgen_core::ml_available(&missing));
}

/// The committed tiny ONNX fixture (`rmvpe_tiny.onnx`) with RMVPE's I/O
/// contract, for the ML end-to-end tests; no real model in CI.
#[cfg(feature = "ml")]
fn ml_fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("kirafrq-ml-provider")
        .join("tests")
        .join("fixtures")
        .join("rmvpe_tiny.onnx")
}

#[cfg(feature = "ml")]
#[test]
fn an_ml_confidence_override_above_every_peak_unvoices_the_whole_file() {
    // #53 criterion 4 end to end: all peaks are below the threshold, so the
    // table is all-unvoiced and its key is `0.0`.
    let scratch = Scratch::new("ml-all-unvoiced");
    write_wav(&scratch, "A2.wav", mono(), &[SAMPLE; 1024]);

    let mut opts = options(&scratch.0, &[Target::Frq]);
    opts.f0.estimator = kirafrqgen_core::Estimator::Rmvpe;
    opts.f0.ml = kirafrqgen_core::MlConfig {
        model_path: Some(ml_fixture()),
        confidence_threshold: Some(2.0),
    };
    let estimator = kirafrqgen_core::build_estimator(&opts.f0, 1).unwrap();
    run(&opts, estimator.as_ref());

    let table = frq::read(&scratch.join("A2_wav.frq")).unwrap();
    assert_eq!(table.f0_hz, vec![0.0; 5], "all-unvoiced");
    assert_eq!(table.key_hz, 0.0);
}

#[cfg(feature = "ml")]
#[test]
fn the_ml_fixture_writes_a_conforming_table() {
    // The factory + pipeline path with the fixture: the #8 conventions hold
    // and the fixture's ~441.5 Hz slot reaches the table.
    let scratch = Scratch::new("ml-fixture");
    write_wav(&scratch, "A2.wav", mono(), &[SAMPLE; 1024]);

    let mut opts = options(&scratch.0, &[Target::Frq]);
    opts.f0.estimator = kirafrqgen_core::Estimator::Rmvpe;
    opts.f0.ml = kirafrqgen_core::MlConfig {
        model_path: Some(ml_fixture()),
        confidence_threshold: None,
    };
    let estimator = kirafrqgen_core::build_estimator(&opts.f0, 1).unwrap();
    assert!(!estimator.supports_stonemask(), "ML never refines");
    run(&opts, estimator.as_ref());

    let table = frq::read(&scratch.join("A2_wav.frq")).unwrap();
    assert_eq!(table.f0_hz.len(), 5);
    assert_eq!(*table.f0_hz.last().unwrap(), 0.0, "the trailing rule");
    let voiced: Vec<f64> = table
        .f0_hz
        .iter()
        .copied()
        .filter(|&value| value > 0.0)
        .collect();
    assert!(!voiced.is_empty(), "{:?}", table.f0_hz);
    assert!(
        voiced.iter().any(|&value| (400.0..500.0).contains(&value)),
        "the fixture's ~441.5 Hz slot: {voiced:?}"
    );
}

#[cfg(feature = "ml")]
#[test]
fn an_unloadable_ml_model_is_an_early_config_error() {
    // #49: a model file that exists but cannot be loaded fails at the factory,
    // before any file is touched.
    let scratch = Scratch::new("ml-unloadable");
    let broken = scratch.join("broken.onnx");
    fs::write(&broken, b"not an onnx graph").unwrap();
    let mut opts = options(&scratch.0, &[Target::Frq]);
    opts.f0.estimator = kirafrqgen_core::Estimator::Rmvpe;
    opts.f0.ml = kirafrqgen_core::MlConfig {
        model_path: Some(broken),
        confidence_threshold: None,
    };
    match kirafrqgen_core::build_estimator(&opts.f0, 1) {
        Err(GeneratorError::Config(message)) => {
            assert!(message.contains("broken.onnx"), "{message}");
        }
        Err(other) => panic!("expected a config error, got {other}"),
        Ok(_) => panic!("expected a config error, got an estimator"),
    }
}

// --- SwiftF0 (#69) ----------------------------------------------------------

#[test]
fn selecting_swiftf0_resolves_the_bundled_model_or_reports_the_missing_feature() {
    let config = kirafrqgen_core::F0Config {
        estimator: kirafrqgen_core::Estimator::SwiftF0,
        ..kirafrqgen_core::F0Config::default()
    };
    let built = kirafrqgen_core::build_estimator(&config, 1);
    if kirafrqgen_core::ML_SUPPORTED {
        let estimator = built.expect("the bundled model always builds in an ML build");
        assert!(!estimator.supports_stonemask(), "ML never refines");
    } else {
        match built {
            Err(GeneratorError::Config(message)) => {
                assert!(message.contains("no ML estimator support"), "{message}");
            }
            Err(other) => panic!("expected a config error, got {other}"),
            Ok(_) => panic!("expected a config error, got an estimator"),
        }
    }
}

#[test]
fn selecting_swiftf0_with_a_missing_explicit_model_is_an_early_config_error() {
    let config = kirafrqgen_core::F0Config {
        estimator: kirafrqgen_core::Estimator::SwiftF0,
        ml: kirafrqgen_core::MlConfig {
            model_path: Some(PathBuf::from("definitely-not-a-model.onnx")),
            confidence_threshold: None,
        },
        ..kirafrqgen_core::F0Config::default()
    };
    match kirafrqgen_core::build_estimator(&config, 1) {
        Err(GeneratorError::Config(message)) => {
            if kirafrqgen_core::ML_SUPPORTED {
                assert!(message.contains("definitely-not-a-model.onnx"), "{message}");
            } else {
                assert!(message.contains("no ML estimator support"), "{message}");
            }
        }
        Err(other) => panic!("expected a config error, got {other}"),
        Ok(_) => panic!("expected a config error, got an estimator"),
    }
}

/// A 44.1 kHz 220 Hz harmonic stack, long enough for a voiced run: #59
/// measured full voicing for this shape.
#[cfg(feature = "ml")]
fn swiftf0_samples(len: usize) -> Vec<i16> {
    (0..len)
        .map(|index| {
            let t = index as f64 / 44_100.0;
            let value: f64 = (1..=10)
                .map(|harmonic| {
                    (std::f64::consts::TAU * 220.0 * harmonic as f64 * t).sin() / harmonic as f64
                })
                .sum();
            (value * 0.1 * 32_767.0) as i16
        })
        .collect()
}

#[cfg(feature = "ml")]
#[test]
fn the_bundled_swiftf0_model_writes_a_conforming_table() {
    // The factory + pipeline path with no model file anywhere: the bundled
    // model must carry a real harmonic through to a conforming table.
    let scratch = Scratch::new("swiftf0-bundled");
    write_wav(&scratch, "A2.wav", mono(), &swiftf0_samples(8192));

    let mut opts = options(&scratch.0, &[Target::Frq]);
    opts.f0.estimator = kirafrqgen_core::Estimator::SwiftF0;
    let estimator = kirafrqgen_core::build_estimator(&opts.f0, 1).unwrap();
    assert!(!estimator.supports_stonemask(), "ML never refines");
    run(&opts, estimator.as_ref());

    let table = frq::read(&scratch.join("A2_wav.frq")).unwrap();
    assert_eq!(table.f0_hz.len(), 8192 / 256 + 1);
    assert_eq!(*table.f0_hz.last().unwrap(), 0.0, "the trailing rule");
    let voiced: Vec<f64> = table
        .f0_hz
        .iter()
        .copied()
        .filter(|&value| value > 0.0)
        .collect();
    assert!(!voiced.is_empty(), "{:?}", table.f0_hz);
    for &value in &voiced {
        assert!(value.is_finite(), "non-finite: {value}");
        assert!((71.0..=800.0).contains(&value), "out of range: {value}");
        assert!((value - 220.0).abs() < 2.0, "not ~220 Hz: {value}");
    }
}

// --- energy voicing gate (#54) ----------------------------------------------

/// 1024 samples: loud, i16 33, loud, loud, empty — frq amplitudes
/// `[8192, 33, 8192, 8192, 0]`, so the second frame sits under the 5% gate
/// and the empty trailing frame is unvoiced by #8 either way.
fn gated_samples() -> Vec<i16> {
    let mut samples = vec![SAMPLE; 1024];
    samples[256..512].fill(33);
    samples
}

#[test]
fn the_energy_gate_forces_quiet_voiced_frames_unvoiced_by_default() {
    let scratch = Scratch::new("energy-gate");
    write_wav(&scratch, "A2.wav", mono(), &gated_samples());
    let estimator = FakeEstimator::new(&[110.0, 220.0, 330.0, 440.0, 550.0]);

    let opts = options(&scratch.0, &[Target::Frq]);
    assert!(opts.f0.world_quirks, "the default is the tuned path");
    run(&opts, &estimator);

    let table = frq::read(&scratch.join("A2_wav.frq")).unwrap();
    assert_eq!(table.f0_hz, [110.0, 0.0, 330.0, 440.0, 0.0]);
    assert_eq!(
        table.amplitude,
        Some(vec![8192.0, 33.0, 8192.0, 8192.0, 0.0])
    );
    assert_eq!(
        table.key_hz,
        880.0 / 3.0,
        "key over the gated voiced frames"
    );
}

#[test]
fn world_quirks_off_writes_the_untouched_estimator_output() {
    let scratch = Scratch::new("energy-gate-off");
    write_wav(&scratch, "A2.wav", mono(), &gated_samples());
    let estimator = FakeEstimator::new(&[110.0, 220.0, 330.0, 440.0, 550.0]);

    let mut opts = options(&scratch.0, &[Target::Frq]);
    opts.f0.world_quirks = false;
    run(&opts, &estimator);

    let table = frq::read(&scratch.join("A2_wav.frq")).unwrap();
    assert_eq!(table.f0_hz, [110.0, 220.0, 330.0, 440.0, 0.0]);
}

#[test]
fn the_energy_gate_applies_when_frq_is_not_a_target() {
    let scratch = Scratch::new("energy-gate-mrq");
    write_wav(&scratch, "A2.wav", mono(), &gated_samples());
    let estimator = FakeEstimator::new(&[110.0, 220.0, 330.0, 440.0, 550.0]);

    run(&options(&scratch.0, &[Target::Mrq]), &estimator);

    let desc = mrq::Desc::read(&mrq::desc_path(&scratch.0))
        .unwrap()
        .unwrap();
    let entry = &desc.entries()[0];
    assert_eq!(entry_nf0(entry), 4);
    assert_eq!(entry_f0(entry), [110.0, 0.0, 330.0, 440.0]);
}

#[test]
fn the_energy_gate_is_a_no_op_when_the_file_is_silent() {
    let scratch = Scratch::new("energy-gate-silence");
    write_wav(&scratch, "A2.wav", mono(), &[0; 512]);
    let estimator = FakeEstimator::new(&[220.0]);

    run(&options(&scratch.0, &[Target::Frq]), &estimator);

    let table = frq::read(&scratch.join("A2_wav.frq")).unwrap();
    assert_eq!(table.f0_hz, [220.0, 220.0, 0.0], "p90 = 0 skips the gate");
    assert!(table.f0_hz.iter().all(|value| value.is_finite()));
}

#[test]
fn the_energy_gate_is_a_no_op_without_voiced_frames() {
    let scratch = Scratch::new("energy-gate-unvoiced");
    write_wav(&scratch, "A2.wav", mono(), &[SAMPLE; 512]);
    let estimator = FakeEstimator::new(&[0.0]);

    run(&options(&scratch.0, &[Target::Frq]), &estimator);

    let table = frq::read(&scratch.join("A2_wav.frq")).unwrap();
    assert_eq!(table.f0_hz, [0.0; 3]);
    assert_eq!(table.key_hz, 0.0);
}

// --- aperiodicity voicing gate (#64) ----------------------------------------

#[test]
fn the_aperiodicity_gate_forces_low_statistic_frames_unvoiced() {
    let scratch = Scratch::new("aperiodicity-gate");
    write_wav(&scratch, "A2.wav", mono(), &[SAMPLE; 1024]);
    let mut estimator = FakeEstimator::new(&[110.0, 220.0, 330.0, 440.0, 550.0]);
    // 0.84 and 0.50 are below the 0.85 split; 0.90 is above it.
    estimator.aperiodicity0 = Some(vec![0.99, 0.84, 0.90, 0.50, 0.99]);

    let opts = options(&scratch.0, &[Target::Frq]);
    assert_eq!(
        opts.f0.estimator,
        kirafrqgen_core::Estimator::Harvest,
        "the default estimator is the Harvest path"
    );
    run(&opts, &estimator);

    let table = frq::read(&scratch.join("A2_wav.frq")).unwrap();
    assert_eq!(table.f0_hz, [110.0, 0.0, 330.0, 0.0, 0.0]);
    assert_eq!(table.key_hz, 220.0, "key over the gated voiced frames");
    assert_eq!(estimator.aperiodicity_calls(), 1);
}

#[test]
fn the_aperiodicity_gate_is_harvest_only() {
    let scratch = Scratch::new("aperiodicity-gate-dio");
    write_wav(&scratch, "A2.wav", mono(), &[SAMPLE; 1024]);
    let mut estimator = FakeEstimator::new(&[110.0, 220.0, 330.0, 440.0, 550.0]);
    // A statistic that would gate every frame: "untouched" is unambiguous.
    estimator.aperiodicity0 = Some(vec![0.0; 5]);

    let mut opts = options(&scratch.0, &[Target::Frq]);
    opts.f0.estimator = kirafrqgen_core::Estimator::Dio;
    assert!(
        opts.f0.world_quirks,
        "the tuned path is on; DIO is excluded"
    );
    run(&opts, &estimator);

    let table = frq::read(&scratch.join("A2_wav.frq")).unwrap();
    assert_eq!(
        table.f0_hz,
        [110.0, 220.0, 330.0, 440.0, 0.0],
        "no aperiodicity gate on DIO"
    );
    assert_eq!(estimator.aperiodicity_calls(), 0, "DIO never asks for D4C");
}

#[test]
fn world_quirks_off_skips_the_aperiodicity_gate() {
    let scratch = Scratch::new("aperiodicity-gate-off");
    write_wav(&scratch, "A2.wav", mono(), &[SAMPLE; 1024]);
    let mut estimator = FakeEstimator::new(&[110.0, 220.0, 330.0, 440.0, 550.0]);
    estimator.aperiodicity0 = Some(vec![0.0; 5]);

    let mut opts = options(&scratch.0, &[Target::Frq]);
    opts.f0.world_quirks = false;
    run(&opts, &estimator);

    let table = frq::read(&scratch.join("A2_wav.frq")).unwrap();
    assert_eq!(table.f0_hz, [110.0, 220.0, 330.0, 440.0, 0.0]);
    assert_eq!(estimator.aperiodicity_calls(), 0);
}

#[test]
fn the_aperiodicity_gate_never_adds_voicing() {
    let scratch = Scratch::new("aperiodicity-gate-no-add");
    write_wav(&scratch, "A2.wav", mono(), &[SAMPLE; 1024]);
    let mut estimator = FakeEstimator::new(&[110.0, 0.0, 330.0, 0.0, 550.0]);
    // High statistics everywhere: the gate never voices an unvoiced frame.
    estimator.aperiodicity0 = Some(vec![1.0; 5]);

    run(&options(&scratch.0, &[Target::Frq]), &estimator);

    let table = frq::read(&scratch.join("A2_wav.frq")).unwrap();
    assert_eq!(table.f0_hz, [110.0, 0.0, 330.0, 0.0, 0.0]);
}

#[test]
fn the_aperiodicity_gate_is_a_no_op_without_voiced_frames() {
    let scratch = Scratch::new("aperiodicity-gate-unvoiced");
    write_wav(&scratch, "A2.wav", mono(), &[SAMPLE; 1024]);
    let mut estimator = FakeEstimator::new(&[0.0]);
    estimator.aperiodicity0 = Some(vec![0.0]);

    run(&options(&scratch.0, &[Target::Frq]), &estimator);

    let table = frq::read(&scratch.join("A2_wav.frq")).unwrap();
    assert_eq!(table.f0_hz, [0.0; 5]);
    assert_eq!(
        estimator.aperiodicity_calls(),
        0,
        "a silence-only / no-voiced file skips the D4C pass"
    );
}

#[test]
fn a_d4c_failure_is_a_per_file_failure() {
    let scratch = Scratch::new("aperiodicity-gate-failure");
    write_wav(&scratch, "A2.wav", mono(), &[SAMPLE; 1024]);
    let mut estimator = FakeEstimator::new(&[220.0]);
    estimator.aperiodicity_error = true;

    let summary = run(&options(&scratch.0, &[Target::Frq]), &estimator);

    assert_eq!(summary.written, 0);
    assert_eq!(summary.failed.len(), 1, "{:?}", summary.failed);
    assert!(summary.failed[0].1.contains("D4C"), "{:?}", summary.failed);
    assert!(!scratch.join("A2_wav.frq").exists());
}

#[test]
fn an_invalid_aperiodicity_gate_threshold_is_a_config_error() {
    let scratch = Scratch::new("aperiodicity-threshold");
    write_wav(&scratch, "A2.wav", mono(), &[SAMPLE; 512]);
    let estimator = FakeEstimator::new(&[220.0]);

    for threshold in [f64::NAN, -0.01] {
        let mut opts = options(&scratch.0, &[Target::Frq]);
        opts.f0.aperiodicity_gate_threshold = threshold;
        assert!(
            matches!(
                run_result(&opts, &estimator),
                Err(GeneratorError::Config(_))
            ),
            "threshold {threshold} must be rejected"
        );
    }
}

// --- progress ---------------------------------------------------------------

#[test]
fn progress_reports_every_file_folder_and_the_summary() {
    let scratch = Scratch::new("progress");
    write_wav(&scratch, "A2/A2.wav", mono(), &[SAMPLE; 512]);
    write_wav(&scratch, "B2/B2.wav", mono(), &[SAMPLE; 512]);

    let estimator = FakeEstimator::new(&[220.0]);
    let opts = options(&scratch.0, &[Target::Mrq]);
    let progress = Recording::default();
    let cancel: CancelToken = Arc::new(AtomicBool::new(false));
    let summary = generate(&opts, &estimator, &progress, &cancel).unwrap();

    assert_eq!(summary.written, 2);
    assert_eq!(progress.started_files.lock().unwrap().len(), 2);
    let finished = progress.finished_files.lock().unwrap();
    assert_eq!(finished.len(), 2);
    assert!(
        finished
            .iter()
            .all(|report| report.written.contains(&Target::Mrq)),
        "file_finished waits for the folder merge"
    );
    let folders = progress.finished_folders.lock().unwrap();
    assert_eq!(folders.len(), 2);
    assert!(folders.contains(&scratch.join("A2")));
    assert!(folders.contains(&scratch.join("B2")));
    assert!(progress.run_summary.lock().unwrap().is_some());
}

// --- frame progress (#34) ---------------------------------------------------

use kirafrq_world_binding::{FrameObserver, ProgressStage};

/// Fake estimator that plays a scripted (stage, done, total) sequence through
/// the observer instead of analyzing anything.
struct ScriptedEstimator {
    /// Whether it advertises StoneMask support; when false, the pipeline never
    /// gives the refine phase a share and the analysis stays the estimate's.
    stone_mask: bool,
}

impl ScriptedEstimator {
    fn new() -> Self {
        Self { stone_mask: true }
    }
}

impl F0Estimator for ScriptedEstimator {
    fn estimate(
        &self,
        samples: &[f64],
        _sample_rate: u32,
        frame_period_ms: f64,
        observer: Option<&dyn FrameObserver>,
    ) -> Result<F0Track, GeneratorError> {
        if let Some(observer) = observer {
            for done in 1..=4 {
                observer.report(ProgressStage::Estimate, done, 4);
            }
        }
        let frames = samples.len() / 256 + 1;
        Ok(F0Track {
            frame_period_ms,
            temporal_positions: (0..frames)
                .map(|index| index as f64 * frame_period_ms / 1000.0)
                .collect(),
            f0_hz: vec![110.0; frames],
        })
    }

    fn refine_stonemask(
        &self,
        _samples: &[f64],
        _sample_rate: u32,
        _track: &mut F0Track,
        observer: Option<&dyn FrameObserver>,
    ) -> Result<(), GeneratorError> {
        if let Some(observer) = observer {
            for done in 1..=4 {
                observer.report(ProgressStage::Refine, done, 4);
            }
        }
        Ok(())
    }

    fn supports_stonemask(&self) -> bool {
        self.stone_mask
    }
}

#[derive(Default)]
struct ProgressTape {
    events: Mutex<Vec<(PathBuf, u64, u64)>>,
}

impl Progress for ProgressTape {
    fn wants_file_progress(&self) -> bool {
        true
    }

    fn file_progress(&self, wav: &Path, done: u64, total: u64) {
        self.events
            .lock()
            .unwrap()
            .push((wav.to_path_buf(), done, total));
    }
}

#[test]
fn file_progress_composes_estimate_and_refine_into_one_permille() {
    let scratch = Scratch::new("file-progress");
    let wav = write_wav(&scratch, "A2.wav", mono(), &[SAMPLE; 1000]);
    let mut opts = options(&scratch.0, &[Target::Frq]);
    opts.f0.stone_mask = true;
    let progress = ProgressTape::default();
    let cancel: CancelToken = Arc::new(AtomicBool::new(false));
    generate(&opts, &ScriptedEstimator::new(), &progress, &cancel).unwrap();

    let events = progress.events.lock().unwrap();
    let fractions: Vec<u64> = events
        .iter()
        .filter(|(path, _, _)| *path == wav)
        .map(|(_, done, _)| *done)
        .collect();
    // The analysis owns the first 90% (estimate the first half of that,
    // refinement the second), latched so the sequence never moves backwards;
    // the terminal 1000 fires once the tables are written.
    assert_eq!(fractions, [112, 225, 337, 450, 562, 675, 787, 900, 1000]);
    assert!(events.iter().all(|(_, _, total)| *total == 1000));
}

#[test]
fn file_progress_without_stonemask_gives_estimate_the_whole_analysis() {
    let scratch = Scratch::new("file-progress-no-refine");
    let wav = write_wav(&scratch, "A2.wav", mono(), &[SAMPLE; 1000]);
    let mut opts = options(&scratch.0, &[Target::Frq]);
    opts.f0.stone_mask = false;
    let progress = ProgressTape::default();
    let cancel: CancelToken = Arc::new(AtomicBool::new(false));
    generate(&opts, &ScriptedEstimator::new(), &progress, &cancel).unwrap();

    let events = progress.events.lock().unwrap();
    let fractions: Vec<u64> = events
        .iter()
        .filter(|(path, _, _)| *path == wav)
        .map(|(_, done, _)| *done)
        .collect();
    assert_eq!(fractions, [225, 450, 675, 900, 1000]);
}

#[test]
fn file_progress_splits_the_write_share_across_targets() {
    let scratch = Scratch::new("file-progress-units");
    let wav = write_wav(&scratch, "A2.wav", mono(), &[SAMPLE; 1000]);
    let opts = options(&scratch.0, &[Target::Frq, Target::Pmk]);
    let progress = ProgressTape::default();
    let cancel: CancelToken = Arc::new(AtomicBool::new(false));
    generate(&opts, &ScriptedEstimator::new(), &progress, &cancel).unwrap();

    let events = progress.events.lock().unwrap();
    let fractions: Vec<u64> = events
        .iter()
        .filter(|(path, _, _)| *path == wav)
        .map(|(_, done, _)| *done)
        .collect();
    // StoneMask is on by default: analysis to 900, then one 50-share per
    // written target.
    assert_eq!(
        fractions,
        [112, 225, 337, 450, 562, 675, 787, 900, 950, 1000]
    );
}

#[test]
fn file_progress_credits_a_deferred_mrq_share_at_the_folder_merge() {
    let scratch = Scratch::new("file-progress-mrq");
    let wav = write_wav(&scratch, "A2.wav", mono(), &[SAMPLE; 1000]);
    let opts = options(&scratch.0, &[Target::Frq, Target::Mrq]);
    let progress = ProgressTape::default();
    let cancel: CancelToken = Arc::new(AtomicBool::new(false));
    generate(&opts, &ScriptedEstimator::new(), &progress, &cancel).unwrap();

    let events = progress.events.lock().unwrap();
    let fractions: Vec<u64> = events
        .iter()
        .filter(|(path, _, _)| *path == wav)
        .map(|(_, done, _)| *done)
        .collect();
    // The frq share lands in phase A; the mrq share waits for the merge.
    assert_eq!(
        fractions,
        [112, 225, 337, 450, 562, 675, 787, 900, 950, 1000]
    );
    assert!(
        events.iter().any(|(_, done, _)| *done == 1000),
        "the merge completes the file"
    );
}

#[test]
fn skipped_wavs_emit_no_progress_events() {
    let scratch = Scratch::new("progress-skip");
    write_wav(&scratch, "A2.wav", mono(), &[SAMPLE; 1000]);
    let estimator = FakeEstimator::new(&[100.0]);
    let opts = options(&scratch.0, &[Target::Frq]);
    let cancel: CancelToken = Arc::new(AtomicBool::new(false));
    let first = ProgressTape::default();
    generate(&opts, &estimator, &first, &cancel).unwrap();

    // Every table exists now, so the second run skips analysis entirely.
    let second = ProgressTape::default();
    generate(&opts, &estimator, &second, &cancel).unwrap();
    assert!(second.events.lock().unwrap().is_empty());
}
