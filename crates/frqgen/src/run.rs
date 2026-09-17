//! Run orchestration: the scan, the parallel per-wav pass, and the per-folder
//! mrq merge.
//!
//! `generate` runs in two phases. Phase A walks the scanned wavs in parallel
//! (a rayon pool sized by `jobs`): decode (#11), estimate, StoneMask, build
//! the table (#8), then apply the per-target sidecar policy. Wavs with an mrq
//! contribution defer their `file_finished` to phase B. Phase B merge-writes
//! each folder's `desc.mrq` once (#10/#16), then finalizes the contributors
//! and reports every folder. Cancellation is checked before decode and before
//! writes; the summary returned is the partial run.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::time::{SystemTime, UNIX_EPOCH};

use frq_core::{FrequencyTable, audio, frq, mrq, pmk};
use rayon::prelude::*;

use crate::paths;
use crate::scan::scan_wavs;
use crate::table::{SAMPLE_RATE, build_table, frame_period_ms};
use crate::{
    CancelToken, F0Estimator, FilePlan, FileReport, GenerateOptions, GeneratorError, Progress,
    RunPlan, RunSummary, Target, llsm,
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
/// Semantics are identical to [`generate`] (including the per-folder mrq
/// merge); `opts.root` is not consulted, so the caller passes the same options
/// it planned with. An empty list is a [`GeneratorError::Config`].
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

    // Phase A: every wav in parallel. A wav with an mrq contribution keeps its
    // report unfinalized until its folder's merge-write in phase B.
    let mut results: Vec<WavResult> = pool.install(|| {
        wavs.par_iter()
            .map(|wav| process_wav(wav, opts, &descs, estimator, progress, cancel))
            .collect()
    });

    // Phase B: group the contributions by folder and merge-write each once.
    let mut contributions: BTreeMap<PathBuf, Vec<usize>> = BTreeMap::new();
    for (index, result) in results.iter().enumerate() {
        if let Some(work) = &result.mrq {
            contributions
                .entry(work.folder.clone())
                .or_default()
                .push(index);
        }
    }
    let folders: BTreeSet<PathBuf> = wavs.iter().map(|wav| folder_of(wav)).collect();
    for folder in &folders {
        match contributions.get(folder) {
            None => {}
            Some(indices) => {
                if cancel.load(Ordering::Relaxed) {
                    for &index in indices {
                        results[index].report.cancelled = true;
                        progress.file_finished(&results[index].report);
                    }
                } else {
                    merge_folder(folder, indices, &mut results, opts, progress);
                }
            }
        }
        progress.folder_finished(folder);
    }

    let summary = summarize(&results);
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

/// A wav's pending mrq contribution: the entries (one per sharing key) built
/// in phase A, upserted by the folder's single merge-write in phase B.
struct MrqWork {
    folder: PathBuf,
    entries: Vec<mrq::Entry>,
}

/// One wav's phase-A product: its report (final except for a deferred mrq
/// contribution) and the pending mrq entries.
struct WavResult {
    report: FileReport,
    mrq: Option<MrqWork>,
}

/// A folder's `desc.mrq` as read before the pass. Corrupt files are noted so
/// `plan` can warn; `generate` recovers them in phase B.
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

/// Process one wav through the whole phase-A pipeline. Fires `file_started`
/// on entry and `file_finished` on exit unless the wav owes an mrq
/// contribution (phase B then owns its final report).
fn process_wav(
    wav: &Path,
    opts: &GenerateOptions,
    descs: &FolderDescs,
    estimator: &dyn F0Estimator,
    progress: &dyn Progress,
    cancel: &CancelToken,
) -> WavResult {
    progress.file_started(wav);
    let mut report = FileReport::new(wav.to_path_buf());

    if cancel.load(Ordering::Relaxed) {
        report.cancelled = true;
        progress.file_finished(&report);
        return WavResult { report, mrq: None };
    }

    // A fill-missing run never touches existing tables, so a wav whose
    // selected targets all exist skips analysis entirely.
    if !opts.overwrite {
        let existing = existing_targets(wav, opts, descs);
        if existing.len() == opts.targets.len() {
            report.existing = existing;
            progress.file_finished(&report);
            return WavResult { report, mrq: None };
        }
    }

    let decoded = match audio::decode_wav(wav) {
        audio::DecodeOutcome::Decoded(decoded) => decoded,
        audio::DecodeOutcome::Empty => {
            report.empty = true;
            progress.file_finished(&report);
            return WavResult { report, mrq: None };
        }
        audio::DecodeOutcome::Failed(reason) => {
            report.failures.push(reason);
            progress.file_finished(&report);
            return WavResult { report, mrq: None };
        }
    };
    report.warnings.extend(decoded.warnings);

    let mut track = match estimator.estimate(&decoded.samples, SAMPLE_RATE, frame_period_ms()) {
        Ok(track) => track,
        Err(error) => {
            report.failures.push(error.to_string());
            progress.file_finished(&report);
            return WavResult { report, mrq: None };
        }
    };
    if opts.f0.stone_mask
        && let Err(error) = estimator.refine_stonemask(&decoded.samples, SAMPLE_RATE, &mut track)
    {
        report.failures.push(format!("StoneMask: {error}"));
        progress.file_finished(&report);
        return WavResult { report, mrq: None };
    }

    let table = build_table(
        &decoded.samples,
        &track,
        opts.targets.contains(&Target::Frq),
    );

    if cancel.load(Ordering::Relaxed) {
        report.cancelled = true;
        progress.file_finished(&report);
        return WavResult { report, mrq: None };
    }

    let mut mrq_work = None;
    for target in &opts.targets {
        match target {
            Target::Frq => write_frq(wav, opts, &table, &mut report),
            Target::Pmk => write_pmk(wav, opts, &table, decoded.samples.len(), &mut report),
            Target::Mrq => mrq_work = write_mrq(wav, opts, &table, descs, &mut report),
        }
    }

    if mrq_work.is_none() {
        progress.file_finished(&report);
    }
    WavResult {
        report,
        mrq: mrq_work,
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

/// The mrq policy (#10): a wav counts as having a table only when all of its
/// entry keys (local and, with the sharing flag, Japanese-side) exist. The
/// entries are built here and merge-written once per folder in phase B; a
/// sub-hop wav produces no entry at all (#8).
fn write_mrq(
    wav: &Path,
    opts: &GenerateOptions,
    table: &FrequencyTable,
    descs: &FolderDescs,
    report: &mut FileReport,
) -> Option<MrqWork> {
    let folder = folder_of(wav);
    let keys = mrq::entry_keys(
        wav.file_name().expect("the scan yields named files"),
        opts.sharing.as_ref(),
    );
    if descs.has_mrq_keys(&folder, &keys) && !opts.overwrite {
        report.existing.insert(Target::Mrq);
        return None;
    }

    let timestamp = mrq::entry_timestamp(SystemTime::now(), wav_mtime(wav));
    let entries: Vec<mrq::Entry> = keys
        .iter()
        .filter_map(|key| mrq::Entry::from_table(key, table, timestamp))
        .collect();
    if entries.is_empty() {
        // Sub-hop wav: `nf0 == 0`, moresampler writes no entry (#8). With
        // `overwrite` a pre-existing stale entry is left verbatim (#10).
        report.no_entry.insert(Target::Mrq);
        return None;
    }
    Some(MrqWork { folder, entries })
}

/// Open the folder's `desc.mrq` for merging (recovering a corrupt file
/// aside), upsert every contribution, write once, then finalize the
/// contributors: mark mrq written and invalidate their `.llsm` caches.
fn merge_folder(
    folder: &Path,
    indices: &[usize],
    results: &mut [WavResult],
    opts: &GenerateOptions,
    progress: &dyn Progress,
) {
    let desc_path = mrq::desc_path(folder);
    match mrq::open_for_merge(&desc_path) {
        Ok(mrq::MergeOutcome { mut desc, state }) => {
            if let mrq::MergeState::Recovered { backup, error } = &state {
                // One folder-level warning for the run's report (#10).
                let warning = format!(
                    "{} is corrupt ({error}); renamed to {}",
                    desc_path.display(),
                    backup.display()
                );
                if let Some(&first) = indices.first() {
                    results[first].report.warnings.push(warning);
                }
            }
            for &index in indices {
                if let Some(work) = results[index].mrq.take() {
                    for entry in work.entries {
                        desc.upsert(entry);
                    }
                }
            }
            match desc.write(&desc_path) {
                Ok(()) => {
                    for &index in indices {
                        let report = &mut results[index].report;
                        report.written.insert(Target::Mrq);
                        delete_llsm(&report.wav, opts, &mut report.warnings);
                    }
                }
                Err(error) => {
                    for &index in indices {
                        results[index]
                            .report
                            .failures
                            .push(format!("{}: {error}", desc_path.display()));
                    }
                }
            }
        }
        Err(error) => {
            for &index in indices {
                results[index].mrq = None;
                results[index]
                    .report
                    .failures
                    .push(format!("{}: {error}", desc_path.display()));
            }
        }
    }
    for &index in indices {
        progress.file_finished(&results[index].report);
    }
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
            // A read error reads as "no entries" here; the merge-write
            // attempt in phase B reports it per file.
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

fn summarize(results: &[WavResult]) -> RunSummary {
    let mut summary = RunSummary {
        considered: results.len(),
        ..RunSummary::default()
    };
    for WavResult { report, .. } in results {
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
