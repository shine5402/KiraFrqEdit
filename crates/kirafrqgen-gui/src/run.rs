//! The run view's state (#23): one row per selected wav, fed from the
//! pipeline's [`Progress`](kirafrqgen_core::Progress) events. Pure state — the
//! eframe layer reads it and the progress bridge writes it.

use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use kirafrqgen_core::{FileReport, RunSummary, Target};

/// What a run row shows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Status {
    Pending,
    /// The pipeline is decoding, estimating or writing. `permille` is the
    /// wav's composed analysis fraction (#34: `Some(0..=1000)`) once frame
    /// events arrive; `None` beforehand, while the icon spins indeterminate.
    Running {
        permille: Option<u32>,
    },
    Written(BTreeSet<Target>),
    Failed(Vec<String>),
    /// Nothing written and nothing failed: the tables existed, or the wav was
    /// zero-length or too short for the only requested entry.
    Skipped {
        /// The wav had zero frames (#11).
        empty: bool,
        /// Targets the wav cannot produce a table for (sub-hop mrq, #8).
        no_entry: BTreeSet<Target>,
    },
    Cancelled,
}

impl Status {
    fn from_report(report: &FileReport) -> Self {
        if !report.failures.is_empty() {
            Status::Failed(report.failures.clone())
        } else if report.cancelled {
            Status::Cancelled
        } else if !report.written.is_empty() {
            Status::Written(report.written.clone())
        } else {
            Status::Skipped {
                empty: report.empty,
                no_entry: report.no_entry.clone(),
            }
        }
    }

    /// No longer pending or running.
    pub fn is_resolved(&self) -> bool {
        !matches!(self, Status::Pending | Status::Running { .. })
    }
}

/// One selected wav's row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Row {
    pub wav: PathBuf,
    pub status: Status,
    /// Non-fatal per-file notes from the report (#11 warnings, `.llsm`
    /// failures, ...); shown on hover.
    pub warnings: Vec<String>,
}

/// The whole run view: rows, the final summary (or a fatal error), and the
/// cancel flag.
#[derive(Debug, Default)]
pub struct RunState {
    rows: Vec<Row>,
    index: HashMap<PathBuf, usize>,
    summary: Option<RunSummary>,
    error: Option<String>,
    cancel_requested: bool,
}

impl RunState {
    /// Rows start [`Status::Pending`] in the caller's order.
    pub fn new(wavs: Vec<PathBuf>) -> Self {
        let index = wavs
            .iter()
            .enumerate()
            .map(|(row, wav)| (wav.clone(), row))
            .collect();
        let rows = wavs
            .into_iter()
            .map(|wav| Row {
                wav,
                status: Status::Pending,
                warnings: Vec::new(),
            })
            .collect();
        Self {
            rows,
            index,
            ..Self::default()
        }
    }

    /// The rows in selection order.
    pub fn rows(&self) -> &[Row] {
        &self.rows
    }

    /// `Progress::file_started`: mark the wav's row running.
    pub fn started(&mut self, wav: &Path) {
        if let Some(&row) = self.index.get(wav) {
            self.rows[row].status = Status::Running { permille: None };
        }
    }

    /// `Progress::file_progress`: latch the running row's fraction. Events
    /// for unknown or already-resolved rows are ignored.
    pub fn progressed(&mut self, wav: &Path, done: u64, total: u64) {
        if total == 0 {
            return;
        }
        let permille = ((done.min(total) * 1000 / total).min(1000)) as u32;
        if let Some(&row) = self.index.get(wav)
            && let Status::Running { permille: slot } = &mut self.rows[row].status
        {
            *slot = Some(permille);
        }
    }

    /// `Progress::file_finished`: the row's final status and warnings.
    pub fn finished_file(&mut self, report: &FileReport) {
        if let Some(&row) = self.index.get(&report.wav) {
            self.rows[row].status = Status::from_report(report);
            self.rows[row].warnings = report.warnings.clone();
        }
    }

    /// `Progress::finished`: the run is over and the summary is final.
    pub fn finished(&mut self, summary: &RunSummary) {
        self.summary = Some(summary.clone());
    }

    /// The run never started or aborted outside the per-file path (scan or
    /// configuration error).
    pub fn aborted(&mut self, error: String) {
        self.error = Some(error);
    }

    /// Marks the cancel button as pressed and disables it; the caller's token
    /// is what stops the pipeline.
    pub fn request_cancel(&mut self) {
        self.cancel_requested = true;
    }

    /// The button already fired; the run may still be finishing its file.
    pub fn cancel_requested(&self) -> bool {
        self.cancel_requested
    }

    /// True until the summary or a fatal error has landed.
    pub fn is_running(&self) -> bool {
        self.summary.is_none() && self.error.is_none()
    }

    /// The final summary, once `Progress::finished` fired.
    pub fn summary(&self) -> Option<&RunSummary> {
        self.summary.as_ref()
    }

    /// The fatal error that stopped the run before any summary.
    pub fn error(&self) -> Option<&str> {
        self.error.as_deref()
    }

    /// Rows no longer pending or running.
    pub fn resolved(&self) -> usize {
        self.rows
            .iter()
            .filter(|row| row.status.is_resolved())
            .count()
    }

    /// The rows with at least one failure — one per wav, unlike
    /// [`RunSummary::failed`], which holds one entry per reason.
    pub fn failed_rows(&self) -> usize {
        self.rows
            .iter()
            .filter(|row| matches!(row.status, Status::Failed(_)))
            .count()
    }

    pub fn total(&self) -> usize {
        self.rows.len()
    }

    /// Overall progress: resolved rows plus the in-flight fraction of each
    /// running row (#34), so the bar moves within a wav; an empty run counts
    /// as complete.
    pub fn fraction(&self) -> f32 {
        if self.rows.is_empty() {
            1.0
        } else {
            let done: f32 = self
                .rows
                .iter()
                .map(|row| match &row.status {
                    Status::Running {
                        permille: Some(permille),
                    } => *permille as f32 / 1000.0,
                    Status::Running { permille: None } | Status::Pending => 0.0,
                    _ => 1.0,
                })
                .sum();
            done / self.rows.len() as f32
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn report(wav: &str) -> FileReport {
        FileReport::new(PathBuf::from(wav))
    }

    fn state(wavs: &[&str]) -> RunState {
        RunState::new(wavs.iter().map(PathBuf::from).collect())
    }

    #[test]
    fn rows_start_pending_and_resolve() {
        let mut state = state(&["/bank/A2.wav", "/bank/A3.wav"]);
        assert_eq!(state.total(), 2);
        assert_eq!(state.resolved(), 0);
        assert_eq!(state.fraction(), 0.0);
        assert!(state.is_running());

        state.started(Path::new("/bank/A2.wav"));
        assert_eq!(
            state.rows()[0].status,
            Status::Running { permille: None },
            "started marks the row running"
        );
        assert_eq!(state.resolved(), 0, "running is not resolved");

        let mut written = report("/bank/A2.wav");
        written.written.insert(Target::Frq);
        state.finished_file(&written);
        assert_eq!(state.resolved(), 1);
        assert_eq!(state.fraction(), 0.5);
        assert_eq!(
            state.rows()[0].status,
            Status::Written(BTreeSet::from([Target::Frq]))
        );
    }

    #[test]
    fn status_maps_the_report() {
        let mut state = state(&["/bank/A2.wav"]);

        let mut skipped = report("/bank/A2.wav");
        skipped.empty = true;
        state.finished_file(&skipped);
        assert_eq!(
            state.rows()[0].status,
            Status::Skipped {
                empty: true,
                no_entry: BTreeSet::new()
            }
        );

        let mut sub_hop = report("/bank/A2.wav");
        sub_hop.no_entry.insert(Target::Mrq);
        state.finished_file(&sub_hop);
        assert_eq!(
            state.rows()[0].status,
            Status::Skipped {
                empty: false,
                no_entry: BTreeSet::from([Target::Mrq])
            }
        );

        let mut cancelled = report("/bank/A2.wav");
        cancelled.cancelled = true;
        state.finished_file(&cancelled);
        assert_eq!(state.rows()[0].status, Status::Cancelled);

        let mut partial = report("/bank/A2.wav");
        partial.written.insert(Target::Pmk);
        partial.failures.push("frq: permission denied".into());
        state.finished_file(&partial);
        assert_eq!(
            state.rows()[0].status,
            Status::Failed(vec!["frq: permission denied".into()]),
            "a partial failure shows as failed"
        );
        assert_eq!(state.failed_rows(), 1, "counted per wav, not per reason");

        let mut warned = report("/bank/A2.wav");
        warned.written.insert(Target::Frq);
        warned.warnings.push("48 kHz input resampled".into());
        state.finished_file(&warned);
        assert_eq!(
            state.rows()[0].status,
            Status::Written(BTreeSet::from([Target::Frq]))
        );
        assert_eq!(state.rows()[0].warnings, ["48 kHz input resampled"]);
    }

    #[test]
    fn events_for_unknown_wavs_are_ignored() {
        let mut state = state(&["/bank/A2.wav"]);
        let mut stray = report("/bank/zz.wav");
        stray.written.insert(Target::Frq);

        state.started(Path::new("/bank/zz.wav"));
        state.finished_file(&stray);

        assert_eq!(state.rows()[0].status, Status::Pending);
        assert_eq!(state.resolved(), 0);
    }

    #[test]
    fn finished_sets_the_summary_and_stops_the_run() {
        let mut state = state(&["/bank/A2.wav"]);
        let summary = RunSummary {
            considered: 1,
            written: 1,
            ..RunSummary::default()
        };

        state.request_cancel();
        state.finished(&summary);

        assert!(state.cancel_requested());
        assert!(!state.is_running());
        assert_eq!(state.summary().unwrap().written, 1);
        assert!(state.error().is_none());
    }

    #[test]
    fn a_fatal_error_stops_the_run_too() {
        let mut state = state(&["/bank/A2.wav"]);
        state.aborted("no wav files found".into());

        assert!(!state.is_running());
        assert_eq!(state.error(), Some("no wav files found"));
    }

    #[test]
    fn an_empty_run_is_complete() {
        let state = RunState::new(Vec::new());
        assert_eq!(state.total(), 0);
        assert_eq!(state.fraction(), 1.0);
    }

    #[test]
    fn in_flight_fractions_move_the_row_and_the_bar() {
        let mut state = state(&["/bank/A2.wav", "/bank/A3.wav"]);
        state.started(Path::new("/bank/A2.wav"));
        state.started(Path::new("/bank/A3.wav"));

        state.progressed(Path::new("/bank/A2.wav"), 500, 1000);
        assert_eq!(
            state.rows()[0].status,
            Status::Running {
                permille: Some(500)
            }
        );
        assert_eq!(state.fraction(), 0.25);

        let mut written = report("/bank/A2.wav");
        written.written.insert(Target::Frq);
        state.finished_file(&written);
        assert_eq!(state.fraction(), 0.5, "a resolved row counts whole");
    }

    #[test]
    fn progress_for_unknown_or_resolved_rows_is_ignored() {
        let mut state = state(&["/bank/A2.wav"]);

        state.progressed(Path::new("/bank/zz.wav"), 500, 1000);
        state.progressed(Path::new("/bank/A2.wav"), 500, 1000);
        assert_eq!(state.rows()[0].status, Status::Pending);
        assert_eq!(state.fraction(), 0.0);

        state.started(Path::new("/bank/A2.wav"));
        state.progressed(Path::new("/bank/A2.wav"), 1, 0);
        assert_eq!(
            state.rows()[0].status,
            Status::Running { permille: None },
            "a zero total carries no fraction"
        );

        state.progressed(Path::new("/bank/A2.wav"), 2000, 1000);
        assert_eq!(
            state.rows()[0].status,
            Status::Running {
                permille: Some(1000)
            },
            "an overshooting count clamps at complete"
        );

        let mut written = report("/bank/A2.wav");
        written.written.insert(Target::Frq);
        state.finished_file(&written);
        state.progressed(Path::new("/bank/A2.wav"), 100, 1000);
        assert!(
            matches!(state.rows()[0].status, Status::Written(_)),
            "late events never reopen a resolved row"
        );
    }
}
