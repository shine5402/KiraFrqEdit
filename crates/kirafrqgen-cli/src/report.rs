//! The stderr report (#12): per-file lines and the closing summary.
//!
//! Every line goes to stderr; stdout stays empty, reserved for a future
//! `--json`. The default view names failing and warning files; `--verbose`
//! adds one line per file including skips and targets; `--quiet` keeps only
//! failures and drops the summary. Completion lines stream as reports arrive,
//! each emitted with a single `eprintln!`, so parallel workers never
//! interleave within a completion line; status-line writes hold the progress
//! lock instead.

use std::collections::BTreeSet;
use std::io::{IsTerminal as _, Write as _};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use kirafrqgen_core::{FilePlan, FileReport, Progress, RunPlan, RunSummary, Target};

use crate::{any_would_write, would_write};

/// How much the report says (#12).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verbosity {
    /// Failures only, no summary.
    Quiet,
    /// Failing and warning files, then the summary.
    Normal,
    /// One line per file, then the summary.
    Verbose,
}

/// How the reporter surfaces live progress (#35).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProgressMode {
    /// No status line: non-TTY stderr, `--quiet`, or `--dry-run`.
    Off,
    /// Default on a TTY: a single in-place `[k/N]` line, stepped per file.
    Overall,
    /// `--progress` on a TTY: `[k/N] <wav> <pct>%`, fed by `file_progress`.
    Detailed,
}

/// The reporter the run drives: prints as reports arrive and the summary when
/// the run ends. Lines are emitted with single `eprintln!` calls, so parallel
/// workers never interleave within a line.
///
/// Live progress (#35) is purely additive: a single in-place status line on
/// stderr, rewritten as the run advances and cleared before every completion
/// line and the summary. When stderr is not a TTY nothing extra prints, so
/// piped logs read exactly as before.
pub struct Reporter {
    verbosity: Verbosity,
    start: Instant,
    mode: ProgressMode,
    state: Mutex<ProgressState>,
}

/// Mutable progress counters, behind a mutex so parallel workers share one
/// status line last-writer-wins.
#[derive(Debug)]
struct ProgressState {
    total: usize,
    finished: usize,
    /// Latest per-wav fraction for `--progress` (wav, done, total).
    current: Option<(PathBuf, u64, u64)>,
    last_render: Option<Instant>,
    /// Permille rendered last; per-frame estimator events below the step are
    /// swallowed.
    last_permille: Option<u64>,
    /// Visible width of the status line, for space-overwrite clearing.
    last_len: usize,
}

impl Reporter {
    pub fn new(verbosity: Verbosity) -> Self {
        Self::build(verbosity, 0, false, false)
    }

    /// A reporter for a real run: `total` is the planned wav count, `detailed`
    /// is `--progress`. The status line enables only on a TTY and never under
    /// `--quiet`; `--dry-run` never builds one (it returns before `generate`).
    pub fn with_progress(verbosity: Verbosity, total: usize, detailed: bool) -> Self {
        let tty = std::io::stderr().is_terminal();
        Self::build(verbosity, total, detailed, tty)
    }

    fn build(verbosity: Verbosity, total: usize, detailed: bool, tty: bool) -> Self {
        let mode = if verbosity == Verbosity::Quiet || !tty {
            ProgressMode::Off
        } else if detailed {
            ProgressMode::Detailed
        } else {
            ProgressMode::Overall
        };
        Self {
            verbosity,
            start: Instant::now(),
            mode,
            state: Mutex::new(ProgressState {
                total,
                finished: 0,
                current: None,
                last_render: None,
                last_permille: None,
                last_len: 0,
            }),
        }
    }

    /// The status line for the current state, if any.
    fn status_line(state: &ProgressState) -> Option<String> {
        match state.current {
            Some((ref wav, done, total)) if total > 0 => Some(detailed_line(
                state.finished,
                state.total,
                wav,
                done,
                total,
            )),
            _ => None,
        }
        .or_else(|| {
            // Overall mode (or detailed before the first event) still shows
            // the bare counter once the run has a total.
            Some(overall_line(state.finished, state.total))
        })
    }

    fn render_locked(&self, state: &mut ProgressState) {
        let Some(line) = Self::status_line(state) else {
            return;
        };
        // Pad short lines so a longer previous frame leaves no ghosts.
        let pad = state.last_len.saturating_sub(line.len());
        eprint!("\r{line}{}", " ".repeat(pad));
        let _ = std::io::stderr().flush();
        state.last_len = line.len();
        state.last_render = Some(Instant::now());
    }

    fn clear_locked(state: &mut ProgressState) {
        if state.last_len > 0 {
            eprint!("\r{}\r", " ".repeat(state.last_len));
            let _ = std::io::stderr().flush();
            state.last_len = 0;
        }
    }
}

impl Progress for Reporter {
    fn wants_file_progress(&self) -> bool {
        self.mode == ProgressMode::Detailed
    }

    fn file_started(&self, wav: &Path) {
        if self.mode != ProgressMode::Detailed {
            return;
        }
        let mut state = self.state.lock().expect("progress lock");
        state.current = Some((wav.to_path_buf(), 0, 1000));
        // A new wav always re-renders: the name changed even when the
        // fraction did not.
        self.render_locked(&mut state);
        state.last_permille = Some(0);
    }

    fn file_progress(&self, wav: &Path, done: u64, total: u64) {
        if self.mode != ProgressMode::Detailed {
            return;
        }
        let now = Instant::now();
        let mut state = self.state.lock().expect("progress lock");
        let wav_changed = state
            .current
            .as_ref()
            .is_none_or(|(current, _, _)| current != wav);
        state.current = Some((wav.to_path_buf(), done, total));
        let permille = permille_of(done, total);
        if should_render_progress(
            state.last_render,
            state.last_permille,
            wav_changed,
            permille,
            now,
        ) {
            self.render_locked(&mut state);
            state.last_permille = Some(permille);
        }
    }

    fn file_finished(&self, report: &FileReport) {
        let completion: Vec<String> = match self.verbosity {
            Verbosity::Quiet => failure_lines(report).to_vec(),
            Verbosity::Normal => problem_lines(report),
            Verbosity::Verbose => vec![file_line(report)],
        };
        if self.mode == ProgressMode::Off {
            for line in &completion {
                eprintln!("{line}");
            }
            return;
        }
        let mut state = self.state.lock().expect("progress lock");
        state.finished = state.finished.saturating_add(1).min(state.total.max(1));
        Self::clear_locked(&mut state);
        for line in &completion {
            eprintln!("{line}");
        }
        // The finished wav is no longer current: drop it so the re-armed line
        // falls back to the overall counter until the next wav's events land.
        // After the last file the line stays cleared for the summary.
        state.current = None;
        state.last_permille = None;
        if state.finished < state.total {
            self.render_locked(&mut state);
        }
    }

    fn finished(&self, summary: &RunSummary) {
        if self.mode != ProgressMode::Off {
            let mut state = self.state.lock().expect("progress lock");
            Self::clear_locked(&mut state);
        }
        if self.verbosity != Verbosity::Quiet {
            eprintln!("{}", summary_line(summary, self.start.elapsed()));
        }
    }
}

/// The default TTY status line: the overall `[k/N]` counter.
pub fn overall_line(finished: usize, total: usize) -> String {
    format!("[{finished}/{total}] files finished")
}

/// The `--progress` status line: the counter plus the current wav and its
/// percent.
pub fn detailed_line(
    finished: usize,
    total: usize,
    wav: &Path,
    done: u64,
    fraction_total: u64,
) -> String {
    let pct = permille_of(done, fraction_total) * 100 / 1000;
    format!("[{finished}/{total}] {} {pct}%", wav.display())
}

/// `done` of `total` in permille, clamped; a zero denominator reads as zero.
fn permille_of(done: u64, total: u64) -> u64 {
    done.min(total)
        .checked_mul(1000)
        .and_then(|scaled| scaled.checked_div(total))
        .unwrap_or(0)
        .min(1000)
}

/// Throttle per-frame estimator events (#35): a new wav always renders, else
/// render once the permille advanced a step or a time slice passed, so a hot
/// frame loop cannot spam the terminal.
fn should_render_progress(
    last_render: Option<Instant>,
    last_permille: Option<u64>,
    wav_changed: bool,
    permille: u64,
    now: Instant,
) -> bool {
    const MIN_PERMILLE_STEP: u64 = 20;
    const MIN_INTERVAL: Duration = Duration::from_millis(100);
    if wav_changed {
        return true;
    }
    let Some(rendered_at) = last_render else {
        return true;
    };
    let last = last_permille.unwrap_or(0);
    if permille.saturating_sub(last) >= MIN_PERMILLE_STEP || last.saturating_sub(permille) > 0 {
        return true;
    }
    now.duration_since(rendered_at) >= MIN_INTERVAL
}

/// The `--verbose` line for one file.
pub fn file_line(report: &FileReport) -> String {
    let mut parts = Vec::new();
    if !report.written.is_empty() {
        parts.push(format!("wrote {}", target_list(&report.written)));
    }
    if !report.existing.is_empty() {
        parts.push(format!(
            "skipped {} (exists)",
            target_list(&report.existing)
        ));
    }
    if !report.no_entry.is_empty() {
        parts.push(format!(
            "skipped {} (no entry)",
            target_list(&report.no_entry)
        ));
    }
    if report.empty {
        parts.push("skipped (empty)".to_string());
    }
    if report.cancelled {
        parts.push("cancelled".to_string());
    }
    for warning in &report.warnings {
        parts.push(format!("warning: {warning}"));
    }
    for failure in &report.failures {
        parts.push(format!("error: {failure}"));
    }
    format!("{}: {}", report.wav.display(), parts.join("; "))
}

/// The default view's lines for one file: the failure line (with any warnings
/// appended) or the warning line; empty when the file was clean.
pub fn problem_lines(report: &FileReport) -> Vec<String> {
    if !report.failures.is_empty() {
        let mut line = failure_line(report);
        if !report.warnings.is_empty() {
            line.push_str(&format!(" (warning: {})", report.warnings.join("; ")));
        }
        vec![line]
    } else if !report.warnings.is_empty() {
        vec![format!(
            "{}: warning: {}",
            report.wav.display(),
            report.warnings.join("; ")
        )]
    } else {
        Vec::new()
    }
}

/// The `--quiet` view's lines for one file: failures only.
pub fn failure_lines(report: &FileReport) -> Vec<String> {
    if report.failures.is_empty() {
        Vec::new()
    } else {
        vec![failure_line(report)]
    }
}

fn failure_line(report: &FileReport) -> String {
    format!(
        "{}: error: {}",
        report.wav.display(),
        report.failures.join("; ")
    )
}

/// The closing summary line; `failed` and `warnings` count distinct files.
pub fn summary_line(summary: &RunSummary, elapsed: Duration) -> String {
    let mut line = format!(
        "considered {}, written {}, skipped {}, failed {}, warnings {}, elapsed {:.1}s",
        summary.considered,
        summary.written,
        summary.skipped,
        distinct_paths(summary.failed.iter().map(|(wav, _)| wav)),
        distinct_paths(summary.warnings.iter().map(|(wav, _)| wav)),
        elapsed.as_secs_f64()
    );
    if summary.cancelled {
        line.push_str(" (cancelled)");
    }
    line
}

/// The dry-run line for one wav under the overwrite policy and sidecar rules.
pub fn plan_line(file: &FilePlan, targets: &BTreeSet<Target>, overwrite: bool) -> String {
    let write: BTreeSet<Target> = targets
        .iter()
        .copied()
        .filter(|target| would_write(file, *target, overwrite))
        .collect();
    let skip: BTreeSet<Target> = targets.difference(&write).copied().collect();
    let mut parts = Vec::new();
    if !write.is_empty() {
        parts.push(format!("would write {}", target_list(&write)));
    }
    if !skip.is_empty() {
        parts.push(format!("would skip {}", target_list(&skip)));
    }
    for warning in &file.warnings {
        parts.push(format!("warning: {warning}"));
    }
    format!("plan: {}: {}", file.wav.display(), parts.join("; "))
}

/// The dry-run summary line.
pub fn plan_summary_line(
    plan: &RunPlan,
    targets: &BTreeSet<Target>,
    overwrite: bool,
    elapsed: Duration,
) -> String {
    let writing = plan
        .files
        .iter()
        .filter(|file| any_would_write(file, targets, overwrite))
        .count();
    let warnings = plan
        .files
        .iter()
        .filter(|file| !file.warnings.is_empty())
        .count();
    format!(
        "plan: considered {}, would write {}, would skip {}, warnings {}, elapsed {:.1}s",
        plan.files.len(),
        writing,
        plan.files.len() - writing,
        warnings,
        elapsed.as_secs_f64()
    )
}

fn target_list(targets: &BTreeSet<Target>) -> String {
    targets
        .iter()
        .map(|target| match target {
            Target::Frq => "frq",
            Target::Pmk => "pmk",
            Target::Mrq => "mrq",
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn distinct_paths<'a>(paths: impl Iterator<Item = &'a std::path::PathBuf>) -> usize {
    paths.collect::<BTreeSet<_>>().len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn report(name: &str) -> FileReport {
        FileReport::new(PathBuf::from(name))
    }

    fn targets(selected: &[Target]) -> BTreeSet<Target> {
        selected.iter().copied().collect()
    }

    #[test]
    fn a_verbose_line_names_writes_skips_and_reasons() {
        let mut written = report("bank/A2.wav");
        written.written = targets(&[Target::Frq]);
        written.warnings = vec!["could not delete cache".to_string()];
        assert_eq!(
            file_line(&written),
            "bank/A2.wav: wrote frq; warning: could not delete cache"
        );

        let mut skipped = report("bank/B2.wav");
        skipped.existing = targets(&[Target::Frq, Target::Pmk]);
        skipped.no_entry = targets(&[Target::Mrq]);
        assert_eq!(
            file_line(&skipped),
            "bank/B2.wav: skipped frq, pmk (exists); skipped mrq (no entry)"
        );

        let mut empty = report("bank/C2.wav");
        empty.empty = true;
        assert_eq!(file_line(&empty), "bank/C2.wav: skipped (empty)");

        let mut failed = report("bank/D2.wav");
        failed.written = targets(&[Target::Pmk]);
        failed.failures = vec!["bank/D2_wav.frq: denied".to_string()];
        assert_eq!(
            file_line(&failed),
            "bank/D2.wav: wrote pmk; error: bank/D2_wav.frq: denied"
        );
    }

    #[test]
    fn problem_lines_are_per_file_and_keep_warnings_beside_failures() {
        let clean = report("bank/A2.wav");
        assert!(problem_lines(&clean).is_empty());

        let mut warned = report("bank/B2.wav");
        warned.warnings = vec!["decode: odd sample rate".to_string()];
        assert_eq!(
            problem_lines(&warned),
            ["bank/B2.wav: warning: decode: odd sample rate"]
        );

        let mut failed = report("bank/C2.wav");
        failed.warnings = vec!["decode: odd sample rate".to_string()];
        failed.failures = vec!["not a wav".to_string(), "truncated".to_string()];
        assert_eq!(
            problem_lines(&failed),
            ["bank/C2.wav: error: not a wav; truncated (warning: decode: odd sample rate)"]
        );
    }

    #[test]
    fn quiet_lines_are_failures_only() {
        let mut warned = report("bank/A2.wav");
        warned.warnings = vec!["decode: odd sample rate".to_string()];
        assert!(failure_lines(&warned).is_empty());

        let mut written = report("bank/B2.wav");
        written.written = targets(&[Target::Frq]);
        assert!(failure_lines(&written).is_empty());

        let mut failed = report("bank/C2.wav");
        failed.failures = vec!["not a wav".to_string()];
        assert_eq!(failure_lines(&failed), ["bank/C2.wav: error: not a wav"]);
    }

    #[test]
    fn the_summary_counts_distinct_files_per_column() {
        let summary = RunSummary {
            considered: 4,
            written: 2,
            skipped: 1,
            failed: vec![
                (PathBuf::from("bank/A2.wav"), "one".to_string()),
                (PathBuf::from("bank/A2.wav"), "two".to_string()),
                (PathBuf::from("bank/B2.wav"), "three".to_string()),
            ],
            warnings: vec![(PathBuf::from("bank/A2.wav"), "w".to_string())],
            cancelled: false,
        };
        assert_eq!(
            summary_line(&summary, Duration::from_millis(1500)),
            "considered 4, written 2, skipped 1, failed 2, warnings 1, elapsed 1.5s"
        );
    }

    #[test]
    fn a_cancelled_summary_says_so() {
        let summary = RunSummary {
            considered: 10,
            written: 1,
            cancelled: true,
            ..RunSummary::default()
        };
        assert!(summary_line(&summary, Duration::from_secs(2)).ends_with(" (cancelled)"));
    }

    #[test]
    fn plan_lines_follow_the_overwrite_policy_per_target() {
        let file = FilePlan {
            wav: PathBuf::from("bank/A2.wav"),
            existing: targets(&[Target::Pmk]),
            warnings: Vec::new(),
        };
        let selected = targets(&[Target::Frq, Target::Pmk]);
        assert_eq!(
            plan_line(&file, &selected, false),
            "plan: bank/A2.wav: would write frq; would skip pmk"
        );
        assert_eq!(
            plan_line(&file, &selected, true),
            "plan: bank/A2.wav: would write frq, pmk"
        );
    }

    #[test]
    fn the_plan_summary_counts_files() {
        let plan = RunPlan {
            files: vec![
                FilePlan {
                    wav: PathBuf::from("bank/A2.wav"),
                    existing: BTreeSet::new(),
                    warnings: Vec::new(),
                },
                FilePlan {
                    wav: PathBuf::from("bank/B2.wav"),
                    existing: targets(&[Target::Frq]),
                    warnings: Vec::new(),
                },
                FilePlan {
                    wav: PathBuf::from("bank/C2.wav"),
                    existing: BTreeSet::new(),
                    warnings: vec!["corrupt desc.mrq".to_string()],
                },
            ],
        };
        assert_eq!(
            plan_summary_line(
                &plan,
                &targets(&[Target::Frq]),
                false,
                Duration::from_millis(30)
            ),
            "plan: considered 3, would write 2, would skip 1, warnings 1, elapsed 0.0s"
        );
    }

    #[test]
    fn overall_lines_carry_the_counter() {
        assert_eq!(overall_line(0, 4), "[0/4] files finished");
        assert_eq!(overall_line(3, 4), "[3/4] files finished");
    }

    #[test]
    fn detailed_lines_name_the_wav_and_percent() {
        assert_eq!(
            detailed_line(1, 4, Path::new("bank/A2.wav"), 0, 1000),
            "[1/4] bank/A2.wav 0%"
        );
        assert_eq!(
            detailed_line(1, 4, Path::new("bank/A2.wav"), 425, 1000),
            "[1/4] bank/A2.wav 42%"
        );
        assert_eq!(
            detailed_line(2, 4, Path::new("bank/B2.wav"), 1000, 1000),
            "[2/4] bank/B2.wav 100%"
        );
    }

    #[test]
    fn per_frame_events_are_throttled() {
        let now = Instant::now();
        // No prior render always renders.
        assert!(should_render_progress(None, None, false, 1, now));
        // A new wav always renders.
        assert!(should_render_progress(
            Some(now),
            Some(500),
            true,
            500,
            now
        ));
        // Tiny advances within the time slice are swallowed.
        assert!(!should_render_progress(
            Some(now),
            Some(500),
            false,
            501,
            now
        ));
        // A 2% step renders even within the slice.
        assert!(should_render_progress(
            Some(now),
            Some(500),
            false,
            520,
            now
        ));
        // The slice passing renders even without advance.
        assert!(should_render_progress(
            Some(now - Duration::from_millis(200)),
            Some(500),
            false,
            500,
            now
        ));
    }

    #[test]
    fn progress_opts_in_only_for_detailed_tty_runs() {
        // Forced TTY: detailed opts into per-file events, overall does not.
        let detailed = Reporter::build(Verbosity::Normal, 3, true, true);
        assert!(detailed.wants_file_progress());
        let overall = Reporter::build(Verbosity::Normal, 3, false, true);
        assert!(!overall.wants_file_progress());
        // Piped stderr or quiet never opts in, even with the flag.
        let piped = Reporter::build(Verbosity::Normal, 3, true, false);
        assert!(!piped.wants_file_progress());
        assert_eq!(piped.mode, ProgressMode::Off);
        let quiet = Reporter::build(Verbosity::Quiet, 3, true, true);
        assert!(!quiet.wants_file_progress());
        assert_eq!(quiet.mode, ProgressMode::Off);
        // The legacy constructor stays silent for piped test output.
        let legacy = Reporter::new(Verbosity::Normal);
        assert_eq!(legacy.mode, ProgressMode::Off);
        assert!(!legacy.wants_file_progress());
    }

    #[test]
    fn the_status_line_clears_and_re_arms_around_completion_output() {
        // Forced TTY, detailed, two files. Status frames go to the test
        // harness's captured stderr; the assertions below are on the state
        // machine: cleared before completion output, re-armed after.
        let reporter = Reporter::build(Verbosity::Normal, 2, true, true);
        let wav = Path::new("bank/A2.wav");
        reporter.file_started(wav);
        {
            let state = reporter.state.lock().unwrap();
            assert!(state.current.is_some());
            assert!(state.last_len > 0, "started renders a frame");
        }
        // A clean file prints no completion line under Normal, but the
        // counter still advances through clear then re-arm.
        reporter.file_finished(&report("bank/A2.wav"));
        {
            let state = reporter.state.lock().unwrap();
            assert_eq!(state.finished, 1);
            assert!(state.current.is_none(), "the finished wav drops out");
            assert!(state.last_len > 0, "the counter re-arms");
            assert_eq!(
                Reporter::status_line(&state).unwrap(),
                "[1/2] files finished"
            );
        }
        reporter.file_finished(&report("bank/B2.wav"));
        {
            let state = reporter.state.lock().unwrap();
            assert_eq!(state.finished, 2);
            assert_eq!(state.last_len, 0, "the last file stays cleared");
        }
        reporter.finished(&RunSummary::default());
        {
            let state = reporter.state.lock().unwrap();
            assert_eq!(state.last_len, 0, "the summary clears the line");
        }

        // Overall mode steps the bare counter the same way.
        let overall = Reporter::build(Verbosity::Normal, 1, false, true);
        overall.file_finished(&report("bank/A2.wav"));
        {
            let state = overall.state.lock().unwrap();
            assert_eq!(state.finished, 1);
            assert_eq!(state.last_len, 0, "a single file stays cleared");
        }
    }
}
