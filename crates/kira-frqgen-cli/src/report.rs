//! The stderr report (#12): per-file lines and the closing summary.
//!
//! Every line goes to stderr; stdout stays empty, reserved for a future
//! `--json`. The default view names failing and warning files; `--verbose`
//! adds one line per file including skips and targets; `--quiet` keeps only
//! failures and drops the summary. Lines stream as reports arrive, and
//! parallel workers never interleave within a line.

use std::collections::BTreeSet;
use std::time::{Duration, Instant};

use kira_frqgen::{FilePlan, FileReport, Progress, RunPlan, RunSummary, Target};

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

/// The reporter the run drives: prints as reports arrive and the summary when
/// the run ends. Lines are emitted with single `eprintln!` calls, so parallel
/// workers never interleave within a line.
pub struct Reporter {
    verbosity: Verbosity,
    start: Instant,
}

impl Reporter {
    pub fn new(verbosity: Verbosity) -> Self {
        Self {
            verbosity,
            start: Instant::now(),
        }
    }
}

impl Progress for Reporter {
    fn file_finished(&self, report: &FileReport) {
        match self.verbosity {
            Verbosity::Quiet => {
                for line in failure_lines(report) {
                    eprintln!("{line}");
                }
            }
            Verbosity::Normal => {
                for line in problem_lines(report) {
                    eprintln!("{line}");
                }
            }
            Verbosity::Verbose => eprintln!("{}", file_line(report)),
        }
    }

    fn finished(&self, summary: &RunSummary) {
        if self.verbosity != Verbosity::Quiet {
            eprintln!("{}", summary_line(summary, self.start.elapsed()));
        }
    }
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
}
