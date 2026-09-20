//! Run orchestration: the scan, the parallel per-wav pass, and the
//! incremental per-folder mrq merge.
//!
//! `generate` walks the scanned wavs in parallel (a rayon pool sized by
//! `jobs`): decode (#11), estimate, StoneMask, build the table (#8), then apply
//! the per-target sidecar policy. `frq` and `pmk` are written straight away; a
//! wav's `mrq` entries are staged into its folder's in-memory `desc.mrq`, and
//! its `file_finished` waits for the write that persists them. The file is
//! rewritten once the folder has accumulated [`MRQ_FLUSH_BATCH`] entries (plus
//! once more at the end for the remainder), and the rows for a batch settle at
//! that write, so rows resolve as the run proceeds without ever claiming a
//! write that is not on disk. Cancellation is checked before decode and before
//! writes; the summary returned is the partial run.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use kirafrq_audio as audio;
use kirafrq_formats::{FrequencyTable, frq, mrq, pmk};
use kirafrq_world_binding::{FrameObserver, ProgressStage};
use rayon::prelude::*;

use crate::paths;
use crate::scan::scan_wavs;
use crate::table::{SAMPLE_RATE, build_table, frame_period_ms};
use crate::voicing;
use crate::{
    CancelToken, Estimator, F0Estimator, FilePlan, FileReport, GenerateOptions, GeneratorError,
    Progress, RunPlan, RunSummary, Target, llsm,
};

/// Run the generation pass over `opts.root` (#7, #8, #10, #11, #12).
///
/// Fatal only for scan/config problems; every per-file problem lands in the
/// returned [`RunSummary`]. On cancellation the partial summary is returned
/// with `cancelled = true`.
pub fn generate(
    opts: &GenerateOptions,
    estimator: &dyn F0Estimator,
    progress: &dyn Progress,
    cancel: &CancelToken,
) -> Result<RunSummary, GeneratorError> {
    // Config before the filesystem, so an invalid run reports the config
    // problem even when the root is bad too.
    validate(opts)?;
    let wavs = scan_wavs(&opts.root)?;
    generate_wavs(opts, &wavs, estimator, progress, cancel)
}

/// Run the generation pass over an explicit wav list instead of a scan.
/// Semantics are identical to [`generate`] (including the incremental
/// per-folder mrq merge); `opts.root` is not consulted, so the caller passes
/// the same options it planned with. An empty list is a
/// [`GeneratorError::Config`].
pub fn generate_wavs(
    opts: &GenerateOptions,
    wavs: &[PathBuf],
    estimator: &dyn F0Estimator,
    progress: &dyn Progress,
    cancel: &CancelToken,
) -> Result<RunSummary, GeneratorError> {
    validate(opts)?;
    if wavs.is_empty() {
        return Err(GeneratorError::Config(
            "no wav files to process".to_string(),
        ));
    }
    let descs = folder_descs(wavs, opts);
    let pool = build_pool(opts.jobs)?;
    let outcomes: Vec<Mutex<FileReport>> = wavs
        .iter()
        .map(|wav| Mutex::new(FileReport::new(wav.clone())))
        .collect();
    let sink = MrqSink::new(&outcomes, opts, progress);

    // Every wav in parallel. A wav's report is finalized as its own pass ends,
    // except an mrq contribution, which waits for the folder write that
    // persists it (a batch flush or the final `finish`).
    pool.install(|| {
        wavs.par_iter()
            .zip(outcomes.par_iter())
            .enumerate()
            .for_each(|(index, (wav, outcome))| {
                process_wav(wav, index, outcome, opts, &descs, &sink, estimator, cancel);
            });
    });

    // Flush each folder's unflushed mrq tail, settle its waiting wavs, and
    // report every folder. This must run before `summarize`, which reads the
    // reports the tail write settles.
    let folders: BTreeSet<PathBuf> = wavs.iter().map(|wav| folder_of(wav)).collect();
    let warnings = sink.finish(&folders);
    let mut summary = summarize(&outcomes);
    summary.warnings.extend(warnings);
    progress.finished(&summary);
    Ok(summary)
}

/// The dry-run path (#12): scan plus existence checks, no decode, analysis or
/// writes. The caller applies the overwrite policy to decide would-write /
/// would-skip per target.
pub fn plan(opts: &GenerateOptions) -> Result<RunPlan, GeneratorError> {
    validate(opts)?;
    let wavs = scan_wavs(&opts.root)?;
    let descs = folder_descs(&wavs, opts);
    let mut warned = BTreeSet::new();
    let files = wavs
        .iter()
        .map(|wav| plan_wav(wav, opts, &descs, &mut warned))
        .collect();
    Ok(RunPlan { files })
}

/// How many staged mrq entries a folder accumulates before its `desc.mrq` is
/// rewritten. Batching keeps a folder from being rewritten once per wav while
/// still flushing before the run ends, so the file follows the run and a crash
/// loses at most this many entries. Counts entries, not wavs, because bytes are
/// the cost that matters (the sharing flag stages two per wav).
const MRQ_FLUSH_BATCH: usize = 64;

/// The run-wide `desc.mrq` accumulator: one [`FolderSink`] per folder, so
/// parallel wavs sharing a folder serialize on their folder's rewrite only,
/// and different folders rewrite independently.
///
/// A wav that stages an entry is not finished here: it joins its folder's
/// pending list and is settled (marked written + `file_finished`) by the write
/// that persists its entry, so a row never claims a write that is not on disk.
/// The pre-allocated reports give the writing thread access to the wavs it
/// settles.
struct MrqSink<'a> {
    folders: Mutex<BTreeMap<PathBuf, FolderSink>>,
    /// One report per wav, in scan order; the writer locks the ones it settles.
    outcomes: &'a [Mutex<FileReport>],
    opts: &'a GenerateOptions,
    progress: &'a dyn Progress,
}

/// One folder's in-memory `desc.mrq` and the wavs waiting on its next write.
#[derive(Default)]
struct FolderSink {
    /// The folder's file, loaded once on the first stage. Rewritten whole
    /// (foreign entries and order preserved, #10).
    desc: mrq::Desc,
    /// Whether [`open`](FolderSink::open) ran; a folder with `error` set never
    /// opens (and never stages).
    opened: bool,
    /// The one-time corrupt-desc warning, handed to the wav that opened it.
    recovered: Option<String>,
    /// An open or write failure that failed this folder; every later wav in it
    /// fails too rather than silently staging into nothing.
    error: Option<String>,
    /// Entries staged since the last write.
    unflushed: usize,
    /// The wavs whose entries are in `unflushed`, in staging order.
    pending: Vec<usize>,
}

impl<'a> MrqSink<'a> {
    fn new(
        outcomes: &'a [Mutex<FileReport>],
        opts: &'a GenerateOptions,
        progress: &'a dyn Progress,
    ) -> Self {
        Self {
            folders: Mutex::new(BTreeMap::new()),
            outcomes,
            opts,
            progress,
        }
    }

    /// Merge one wav's entries into its folder's file. The folder is rewritten
    /// and its pending wavs settled once it crosses [`MRQ_FLUSH_BATCH`];
    /// otherwise the wav waits for a later write or [`finish`](MrqSink::finish).
    /// A folder-level open failure settles the wav as failed here.
    fn stage(&self, folder: &Path, index: usize, entries: Vec<mrq::Entry>) {
        let mut map = self.folders.lock().expect("the mrq sink is not poisoned");
        let sink = map.entry(folder.to_path_buf()).or_default();
        if let Some(error) = sink.error.clone() {
            settle_failed(&self.outcomes[index], &error, self.progress);
            return;
        }
        if !sink.opened
            && let Err(error) = sink.open(folder)
        {
            settle_failed(&self.outcomes[index], &error, self.progress);
            return;
        }
        if let Some(warning) = sink.recovered.take() {
            self.outcomes[index]
                .lock()
                .expect("the outcome is not poisoned")
                .warnings
                .push(warning);
        }
        for entry in entries {
            sink.desc.upsert(entry);
            sink.unflushed += 1;
        }
        sink.pending.push(index);
        if sink.unflushed >= MRQ_FLUSH_BATCH {
            settle_folder(folder, sink, self.outcomes, self.opts, self.progress);
        }
    }

    /// Write every folder's unflushed tail, settle its waiting wavs and report
    /// each folder. The tail is written even after a cancellation, so entries
    /// already staged for it land.
    fn finish(&self, folders: &BTreeSet<PathBuf>) -> Vec<(PathBuf, String)> {
        let mut warnings = Vec::new();
        let mut sinks = self.folders.lock().expect("the mrq sink is not poisoned");
        for folder in folders {
            if let Some(sink) = sinks.get_mut(folder) {
                if sink.error.is_some() {
                    settle_pending_failed(sink, self.outcomes, self.progress);
                } else if sink.unflushed > 0
                    && let Some(warning) =
                        settle_folder(folder, sink, self.outcomes, self.opts, self.progress)
                {
                    warnings.push((folder.clone(), warning));
                }
            }
            self.progress.folder_finished(folder);
        }
        warnings
    }
}

impl FolderSink {
    /// Open the folder's file once: a corrupt/unsupported one is renamed aside
    /// by [`mrq::open_for_merge`], noted as a warning for the first wav, and
    /// replaced by a fresh v2. A read failure fails the folder.
    fn open(&mut self, folder: &Path) -> Result<(), String> {
        let desc_path = mrq::desc_path(folder);
        match mrq::open_for_merge(&desc_path) {
            Ok(outcome) => {
                self.desc = outcome.desc;
                if let mrq::MergeState::Recovered { backup, error } = outcome.state {
                    self.recovered = Some(format!(
                        "{} is corrupt ({error}); renamed to {}",
                        desc_path.display(),
                        backup.display()
                    ));
                }
                self.opened = true;
                Ok(())
            }
            Err(error) => Err(self.fail(format!("{}: {error}", desc_path.display()))),
        }
    }

    /// Rewrite the whole folder file from the in-memory [`Desc`](mrq::Desc); a
    /// success clears the unflushed count, a failure latches the folder.
    fn write(&mut self, folder: &Path) -> Result<(), String> {
        let desc_path = mrq::desc_path(folder);
        match self.desc.write(&desc_path) {
            Ok(()) => {
                self.unflushed = 0;
                Ok(())
            }
            Err(error) => Err(self.fail(format!("{}: {error}", desc_path.display()))),
        }
    }

    /// Latch a folder-level failure and hand back the message for this wav.
    fn fail(&mut self, message: String) -> String {
        self.error = Some(message.clone());
        message
    }
}

/// Write the folder file, then settle its pending wavs: `written` on success,
/// failed on a write error. Returns the failure message for the run report.
fn settle_folder(
    folder: &Path,
    sink: &mut FolderSink,
    outcomes: &[Mutex<FileReport>],
    opts: &GenerateOptions,
    progress: &dyn Progress,
) -> Option<String> {
    match sink.write(folder) {
        Ok(()) => {
            for index in sink.pending.drain(..) {
                settle_written(&outcomes[index], opts, progress);
            }
            None
        }
        Err(message) => {
            for index in sink.pending.drain(..) {
                settle_failed(&outcomes[index], &message, progress);
            }
            Some(message)
        }
    }
}

/// Settle any wavs still waiting on an already-failed folder.
fn settle_pending_failed(
    sink: &mut FolderSink,
    outcomes: &[Mutex<FileReport>],
    progress: &dyn Progress,
) {
    let message = sink
        .error
        .clone()
        .unwrap_or_else(|| "desc.mrq write failed".to_string());
    for index in sink.pending.drain(..) {
        settle_failed(&outcomes[index], &message, progress);
    }
}

/// Mark a wav's mrq entry written (its folder write landed), drop its `.llsm`
/// cache and fire `file_finished`.
fn settle_written(outcome: &Mutex<FileReport>, opts: &GenerateOptions, progress: &dyn Progress) {
    let snapshot = {
        let mut report = outcome.lock().expect("the outcome is not poisoned");
        report.written.insert(Target::Mrq);
        let wav = report.wav.clone();
        delete_llsm(&wav, opts, &mut report.warnings);
        report.clone()
    };
    progress.file_progress(&snapshot.wav, 1000, 1000);
    progress.file_finished(&snapshot);
}

/// Fail a wav whose mrq entry could not be persisted and fire `file_finished`.
fn settle_failed(outcome: &Mutex<FileReport>, message: &str, progress: &dyn Progress) {
    let snapshot = {
        let mut report = outcome.lock().expect("the outcome is not poisoned");
        report.failures.push(message.to_string());
        report.clone()
    };
    progress.file_finished(&snapshot);
}

/// Apply `edit` to a wav's report and fire `file_finished`: the wav is settled
/// without a deferred mrq contribution.
fn settle_with(
    outcome: &Mutex<FileReport>,
    progress: &dyn Progress,
    edit: impl FnOnce(&mut FileReport),
) {
    let snapshot = {
        let mut report = outcome.lock().expect("the outcome is not poisoned");
        edit(&mut report);
        report.clone()
    };
    progress.file_finished(&snapshot);
}

/// A folder's `desc.mrq` as read before the pass. Corrupt files are noted so
/// `plan` can warn; `generate` recovers them on the first stage.
#[derive(Debug, Default)]
struct FolderDescs {
    loaded: BTreeMap<PathBuf, mrq::Desc>,
    corrupt: BTreeSet<PathBuf>,
}

impl FolderDescs {
    /// Whether the folder's `desc.mrq` holds an entry for every key the wav
    /// needs (#10): with the sharing flag on, a one-sided entry counts as
    /// absent.
    fn has_mrq_keys(&self, folder: &Path, keys: &[Vec<u16>]) -> bool {
        self.loaded
            .get(folder)
            .is_some_and(|desc| keys.iter().all(|key| desc.has(key)))
    }
}

/// One wav's frame-progress observer (#34): composes the estimate and refine
/// phases into a single permille fraction so the indicator never jumps back
/// when refinement starts. Each phase is latched at its high-water mark; the
/// analysis owns the first 90% of the file (estimation the first half of
/// that, refinement the second), and the write phase owns the last 10%: the
/// pipeline emits the terminal 1000 once the wav's tables are written.
struct FileObserver<'a> {
    progress: &'a dyn Progress,
    wav: PathBuf,
    refine: bool,
    estimate: AtomicU64,
    refined: AtomicU64,
}

impl FrameObserver for FileObserver<'_> {
    fn report(&self, stage: ProgressStage, done: usize, total: usize) {
        if total == 0 {
            return;
        }
        let permille = (done.min(total) as u64 * 1000 / total as u64).min(1000);
        let slot = match stage {
            ProgressStage::Estimate => &self.estimate,
            ProgressStage::Refine => &self.refined,
        };
        slot.fetch_max(permille, Ordering::Relaxed);
        let estimate = self.estimate.load(Ordering::Relaxed);
        let done = if self.refine {
            (estimate + self.refined.load(Ordering::Relaxed)) * 9 / 20
        } else {
            estimate * 9 / 10
        };
        self.progress.file_progress(&self.wav, done, 1000);
    }
}

/// Process one wav through the whole pipeline into its pre-allocated report.
/// `file_started` fires on entry. A wav with an mrq entry is handed to the
/// sink, which fires `file_finished` when the write that persists the entry
/// lands; every other wav settles here.
#[allow(clippy::too_many_arguments)]
fn process_wav(
    wav: &Path,
    index: usize,
    outcome: &Mutex<FileReport>,
    opts: &GenerateOptions,
    descs: &FolderDescs,
    sink: &MrqSink,
    estimator: &dyn F0Estimator,
    cancel: &CancelToken,
) {
    let progress = sink.progress;
    progress.file_started(wav);

    if cancel.load(Ordering::Relaxed) {
        settle_with(outcome, progress, |report| report.cancelled = true);
        return;
    }

    // A fill-missing run never touches existing tables, so a wav whose
    // selected targets all exist skips analysis entirely.
    if !opts.overwrite {
        let existing = existing_targets(wav, opts, descs);
        if existing.len() == opts.targets.len() {
            settle_with(outcome, progress, |report| report.existing = existing);
            return;
        }
    }

    let decoded = match audio::decode_wav(wav) {
        audio::DecodeOutcome::Decoded(decoded) => decoded,
        audio::DecodeOutcome::Empty => {
            settle_with(outcome, progress, |report| report.empty = true);
            return;
        }
        audio::DecodeOutcome::Failed(reason) => {
            settle_with(outcome, progress, |report| report.failures.push(reason));
            return;
        }
    };
    outcome
        .lock()
        .expect("the outcome is not poisoned")
        .warnings
        .extend(decoded.warnings);

    // One observer across both phases: the estimate latch carries into the
    // refine half, so the composed fraction never jumps back. Reporters that
    // ignore progress skip the hook entirely (#34). An estimator without
    // StoneMask (#48) never has a refine half.
    let refine = opts.f0.stone_mask && estimator.supports_stonemask();
    let observer = progress.wants_file_progress().then(|| FileObserver {
        progress,
        wav: wav.to_path_buf(),
        refine,
        estimate: AtomicU64::new(0),
        refined: AtomicU64::new(0),
    });
    let probe = observer
        .as_ref()
        .map(|observer| observer as &dyn FrameObserver);

    let mut track =
        match estimator.estimate(&decoded.samples, SAMPLE_RATE, frame_period_ms(), probe) {
            Ok(track) => track,
            Err(error) => {
                settle_with(outcome, progress, |report| {
                    report.failures.push(error.to_string());
                });
                return;
            }
        };
    if refine
        && let Err(error) =
            estimator.refine_stonemask(&decoded.samples, SAMPLE_RATE, &mut track, probe)
    {
        settle_with(outcome, progress, |report| {
            report.failures.push(format!("StoneMask: {error}"));
        });
        return;
    }

    // The recommended tuning's post-passes (#54/#64/#70): the energy gate
    // covers every estimator except RMVPE, whose model confidence is its own
    // voicing policy (#53), and the aperiodicity gate stays Harvest-only.
    if opts.f0.recommended_tuning {
        if opts.f0.estimator.supports_energy_gate() {
            voicing::apply_energy_gate(&decoded.samples, &mut track, opts.f0.energy_gate_ratio);
        }

        // A no-voiced (silence-only) track is a no-op and skips the D4C pass
        // entirely (#64).
        if opts.f0.estimator == Estimator::Harvest && voicing::has_voiced(&track) {
            match estimator.aperiodicity0(&decoded.samples, SAMPLE_RATE, &track) {
                Ok(Some(statistic)) => voicing::apply_aperiodicity_gate(
                    &mut track,
                    &statistic,
                    opts.f0.aperiodicity_gate_threshold,
                ),
                Ok(None) => {}
                Err(error) => {
                    settle_with(outcome, progress, |report| {
                        report.failures.push(format!("D4C: {error}"));
                    });
                    return;
                }
            }
        }
    }

    let table = build_table(
        &decoded.samples,
        &track,
        opts.targets.contains(&Target::Frq),
    );

    if cancel.load(Ordering::Relaxed) {
        settle_with(outcome, progress, |report| report.cancelled = true);
        return;
    }

    // The write phase owns the last 10% of the file, one share per selected
    // target. `frq` and `pmk` credit here; an mrq entry is deferred to the
    // sink, so its share credits with the write that persists it.
    let total_units = opts.targets.len() as u64;
    let mut completed_units = 0u64;
    let mut deferred: Option<Vec<mrq::Entry>> = None;
    {
        let mut report = outcome.lock().expect("the outcome is not poisoned");
        for target in &opts.targets {
            match target {
                Target::Frq => write_frq(wav, opts, &table, &mut report),
                Target::Pmk => write_pmk(wav, opts, &table, decoded.samples.len(), &mut report),
                Target::Mrq => match plan_mrq(wav, opts, &table, descs) {
                    MrqPlan::Existing => {
                        report.existing.insert(Target::Mrq);
                    }
                    MrqPlan::NoEntry => {
                        report.no_entry.insert(Target::Mrq);
                    }
                    MrqPlan::Entries(entries) => deferred = Some(entries),
                },
            }
            if *target != Target::Mrq || deferred.is_none() {
                completed_units += 1;
                progress.file_progress(wav, 900 + completed_units * 100 / total_units, 1000);
            }
        }
    }

    match deferred {
        Some(entries) => sink.stage(&folder_of(wav), index, entries),
        None => {
            let snapshot = outcome.lock().expect("the outcome is not poisoned").clone();
            progress.file_finished(&snapshot);
        }
    }
}

/// The frq policy (#12): fill missing by default; with `overwrite`, rewrite
/// the canonical table and — only after a successful write — delete the
/// alternate spelling.
fn write_frq(wav: &Path, opts: &GenerateOptions, table: &FrequencyTable, report: &mut FileReport) {
    let canonical = paths::canonical_table_path(wav, "frq");
    let alternate = paths::alternate_frq_path(wav);
    if !opts.overwrite && paths::frq_exists(wav) {
        report.existing.insert(Target::Frq);
        return;
    }
    match frq::write(&canonical, table) {
        Ok(()) => {
            report.written.insert(Target::Frq);
            if opts.overwrite
                && alternate.exists()
                && let Err(error) = fs::remove_file(&alternate)
            {
                report
                    .warnings
                    .push(format!("could not delete {}: {error}", alternate.display()));
            }
        }
        Err(error) => report
            .failures
            .push(format!("{}: {error}", canonical.display())),
    }
}

/// The pmk policy (#12): the canonical spelling only; pmk carries no f0
/// invalidation, so `.llsm` is untouched.
fn write_pmk(
    wav: &Path,
    opts: &GenerateOptions,
    table: &FrequencyTable,
    length_samples: usize,
    report: &mut FileReport,
) {
    let canonical = paths::canonical_table_path(wav, "pmk");
    if !opts.overwrite && paths::pmk_exists(wav) {
        report.existing.insert(Target::Pmk);
        return;
    }
    match pmk::write(&canonical, table, length_samples) {
        Ok(()) => {
            report.written.insert(Target::Pmk);
        }
        Err(error) => report
            .failures
            .push(format!("{}: {error}", canonical.display())),
    }
}

/// What one wav's mrq target resolves to.
enum MrqPlan {
    /// Every entry key already exists (fill-missing), so nothing is staged.
    Existing,
    /// A sub-hop wav: no entry is possible (#8).
    NoEntry,
    /// The entries to stage into the folder's pending merge.
    Entries(Vec<mrq::Entry>),
}

/// The mrq policy (#10): a wav counts as having a table only when all of its
/// entry keys (local and, with the sharing flag, the Japanese-side key) exist,
/// and a sub-hop wav produces no entry (#8). The entries are returned, not
/// written: the sink stages them and owns the wav's `file_finished`.
fn plan_mrq(
    wav: &Path,
    opts: &GenerateOptions,
    table: &FrequencyTable,
    descs: &FolderDescs,
) -> MrqPlan {
    let folder = folder_of(wav);
    let keys = mrq::entry_keys(
        wav.file_name().expect("the scan yields named files"),
        opts.sharing.as_ref(),
    );
    if descs.has_mrq_keys(&folder, &keys) && !opts.overwrite {
        return MrqPlan::Existing;
    }

    let timestamp = mrq::entry_timestamp(SystemTime::now(), wav_mtime(wav));
    let entries: Vec<mrq::Entry> = keys
        .iter()
        .filter_map(|key| mrq::Entry::from_table(key, table, timestamp))
        .collect();
    if entries.is_empty() {
        // Sub-hop wav: `nf0 == 0`, moresampler writes no entry (#8). With
        // `overwrite` a pre-existing stale entry is left verbatim (#10).
        return MrqPlan::NoEntry;
    }
    MrqPlan::Entries(entries)
}

/// Read each folder's `desc.mrq` once; a folder without one (or with an
/// unreadable one) simply has no entries for the existence checks.
fn folder_descs(wavs: &[PathBuf], opts: &GenerateOptions) -> FolderDescs {
    let mut descs = FolderDescs::default();
    if !opts.targets.contains(&Target::Mrq) {
        return descs;
    }
    for wav in wavs {
        let folder = folder_of(wav);
        if descs.loaded.contains_key(&folder) || descs.corrupt.contains(&folder) {
            continue;
        }
        match mrq::Desc::read(&mrq::desc_path(&folder)) {
            Ok(Some(desc)) => {
                descs.loaded.insert(folder, desc);
            }
            // A read error reads as "no entries" here; the first stage's open
            // attempt reports it per file.
            Ok(None) | Err(mrq::ReadError::Io(_)) => {}
            Err(mrq::ReadError::Parse(_)) => {
                descs.corrupt.insert(folder);
            }
        }
    }
    descs
}

/// The selected targets whose table already exists per the sidecar rules
/// (#12/#10): frq canonical or alternate, pmk canonical, and mrq only when
/// every sharing key is present in the folder's `desc.mrq`.
fn existing_targets(wav: &Path, opts: &GenerateOptions, descs: &FolderDescs) -> BTreeSet<Target> {
    let mut existing = BTreeSet::new();
    for target in &opts.targets {
        match target {
            Target::Frq if paths::frq_exists(wav) => {
                existing.insert(Target::Frq);
            }
            Target::Pmk if paths::pmk_exists(wav) => {
                existing.insert(Target::Pmk);
            }
            Target::Mrq => {
                let keys = mrq::entry_keys(
                    wav.file_name().expect("the scan yields named files"),
                    opts.sharing.as_ref(),
                );
                if descs.has_mrq_keys(&folder_of(wav), &keys) {
                    existing.insert(Target::Mrq);
                }
            }
            _ => {}
        }
    }
    existing
}

/// Plan one wav. The corrupt-folder note is per folder (#10), so only the
/// first wav of a corrupt folder carries it.
fn plan_wav(
    wav: &Path,
    opts: &GenerateOptions,
    descs: &FolderDescs,
    warned: &mut BTreeSet<PathBuf>,
) -> FilePlan {
    let mut warnings = Vec::new();
    let folder = folder_of(wav);
    if descs.corrupt.contains(&folder) && warned.insert(folder.clone()) {
        warnings.push(format!(
            "{} is corrupt; it will be renamed aside and rebuilt",
            mrq::desc_path(&folder).display()
        ));
    }
    FilePlan {
        wav: wav.to_path_buf(),
        existing: existing_targets(wav, opts, descs),
        warnings,
    }
}

fn summarize(outcomes: &[Mutex<FileReport>]) -> RunSummary {
    let mut summary = RunSummary {
        considered: outcomes.len(),
        ..RunSummary::default()
    };
    for outcome in outcomes {
        let report = outcome.lock().expect("the outcome is not poisoned");
        let report = &*report;
        if report.cancelled {
            summary.cancelled = true;
        }
        if !report.written.is_empty() {
            summary.written += 1;
        }
        if report.skipped() {
            summary.skipped += 1;
        }
        for failure in &report.failures {
            summary.failed.push((report.wav.clone(), failure.clone()));
        }
        for warning in &report.warnings {
            summary.warnings.push((report.wav.clone(), warning.clone()));
        }
    }
    summary
}

fn delete_llsm(wav: &Path, opts: &GenerateOptions, warnings: &mut Vec<String>) {
    if !opts.delete_llsm {
        return;
    }
    if let Err(error) = llsm::delete_cache(wav) {
        warnings.push(format!(
            "could not delete {}: {error}",
            llsm::cache_path(wav).display()
        ));
    }
}

/// #10's `timestamp = max(now, floor(wav_mtime))`; a metadata failure falls
/// back to the epoch, so `now` wins.
fn wav_mtime(wav: &Path) -> SystemTime {
    fs::metadata(wav)
        .and_then(|metadata| metadata.modified())
        .unwrap_or(UNIX_EPOCH)
}

fn folder_of(wav: &Path) -> PathBuf {
    wav.parent().map(Path::to_path_buf).unwrap_or_default()
}

fn validate(opts: &GenerateOptions) -> Result<(), GeneratorError> {
    if opts.targets.is_empty() {
        return Err(GeneratorError::Config(
            "no table formats selected".to_string(),
        ));
    }
    let f0 = &opts.f0;
    if !f0.floor_hz.is_finite()
        || !f0.ceiling_hz.is_finite()
        || f0.floor_hz <= 0.0
        || f0.ceiling_hz <= f0.floor_hz
    {
        return Err(GeneratorError::Config(
            "f0 floor and ceiling must be finite with 0 < floor < ceiling".to_string(),
        ));
    }
    if !f0.energy_gate_ratio.is_finite() || f0.energy_gate_ratio < 0.0 {
        return Err(GeneratorError::Config(
            "the energy gate ratio must be finite and non-negative".to_string(),
        ));
    }
    if !f0.aperiodicity_gate_threshold.is_finite() || f0.aperiodicity_gate_threshold < 0.0 {
        return Err(GeneratorError::Config(
            "the aperiodicity gate threshold must be finite and non-negative".to_string(),
        ));
    }
    Ok(())
}

fn build_pool(jobs: usize) -> Result<rayon::ThreadPool, GeneratorError> {
    let builder = rayon::ThreadPoolBuilder::new();
    let builder = if jobs == 0 {
        builder
    } else {
        builder.num_threads(jobs)
    };
    builder
        .build()
        .map_err(|error| GeneratorError::Config(format!("cannot create the worker pool: {error}")))
}
