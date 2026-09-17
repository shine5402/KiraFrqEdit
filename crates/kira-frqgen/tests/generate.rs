//! Contract for `kira_frqgen::generate`, `kira_frqgen::generate_wavs` and
//! `kira_frqgen::plan` over the settled policy set, exercised on real wavs in
//! a scratch folder with a fake [`F0Estimator`] (no WORLD).

use std::collections::BTreeSet;
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use hound::{SampleFormat, WavSpec, WavWriter};
use kira_frq_core::{FrequencyTable, frq, mrq};
use kira_frqgen::{
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
            "kira-frqgen-{tag}-{}-{}",
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
}

impl FakeEstimator {
    fn new(pattern: &[f64]) -> Self {
        Self {
            pattern: pattern.to_vec(),
            calls: Mutex::new(Vec::new()),
            refined: AtomicUsize::new(0),
            refine_factor: 1.0,
            cancel_on_call: None,
        }
    }

    fn calls(&self) -> Vec<Call> {
        self.calls.lock().unwrap().clone()
    }

    fn refined_count(&self) -> usize {
        self.refined.load(Ordering::SeqCst)
    }
}

impl F0Estimator for FakeEstimator {
    fn estimate(
        &self,
        samples: &[f64],
        sample_rate: u32,
        frame_period_ms: f64,
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
    ) -> Result<(), GeneratorError> {
        self.refined.fetch_add(1, Ordering::SeqCst);
        for value in &mut track.f0_hz {
            if *value > 0.0 {
                *value *= self.refine_factor;
            }
        }
        Ok(())
    }
}

// --- run plumbing -----------------------------------------------------------

fn options(root: &Path, targets: &[Target]) -> GenerateOptions {
    GenerateOptions {
        root: root.to_path_buf(),
        targets: targets.iter().copied().collect(),
        overwrite: false,
        f0: kira_frqgen::F0Config::default(),
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
