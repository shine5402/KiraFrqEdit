//! The `kirafrqgen-cli` command-line interface (#12): `args` owns the clap
//! surface, `prompt` the single interactive question's policy, `report` the
//! stderr lines and `serve` the ML server mode (#85); `main.rs` wires them to
//! the generation pipeline.

pub mod args;
pub mod prompt;
pub mod report;
#[cfg(feature = "ml")]
pub mod serve;

use std::collections::BTreeSet;

use kirafrqgen_core::{F0Config, FilePlan, RunSummary, Target};

/// With the remote runner, ask the ML server before anything runs (#85): an
/// unreachable server, a protocol mismatch or a missing or wrong token is an
/// `Err`. Without an explicit `--estimator`, the default follows the
/// server's availability (RMVPE when the server has its model). Returns the
/// warning for a server on another KiraFrqGen version. A DSP run never needs
/// the server.
pub fn resolve_remote(
    explicit_estimator: bool,
    f0: &mut F0Config,
) -> Result<Option<String>, String> {
    use kirafrqgen_core::Estimator;

    if !f0.ml.runner.is_remote()
        || (explicit_estimator && matches!(f0.estimator, Estimator::Dio | Estimator::Harvest))
    {
        return Ok(None);
    }
    let info = kirafrqgen_core::ml_server_info(&f0.ml)?;
    if !explicit_estimator {
        f0.estimator = kirafrqgen_core::default_estimator(info.rmvpe);
    }
    Ok(info.version_warning)
}

/// Whether `target` would be written for `file` under the run's overwrite
/// policy (#12): a missing table, or every selected table when overwriting.
pub fn would_write(file: &FilePlan, target: Target, overwrite: bool) -> bool {
    overwrite || !file.existing.contains(&target)
}

/// Whether any of `targets` would be written for `file`.
pub fn any_would_write(file: &FilePlan, targets: &BTreeSet<Target>, overwrite: bool) -> bool {
    targets
        .iter()
        .any(|target| would_write(file, *target, overwrite))
}

/// #12's exit code for a completed run: `0` clean — warnings, skips and
/// "nothing to do" included — `1` any per-file failure, `130` cancelled (a
/// partial summary was printed).
pub fn exit_code(summary: &RunSummary) -> u8 {
    if summary.cancelled {
        130
    } else if summary.failed.is_empty() {
        0
    } else {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn file(existing: &[Target]) -> FilePlan {
        FilePlan {
            wav: PathBuf::from("bank/A2.wav"),
            existing: existing.iter().copied().collect(),
            outdated: BTreeSet::new(),
            warnings: Vec::new(),
        }
    }

    #[test]
    fn overwrite_writes_existing_and_missing_tables_alike() {
        let existing = file(&[Target::Frq]);
        assert!(!would_write(&existing, Target::Frq, false));
        assert!(would_write(&existing, Target::Frq, true));
        assert!(would_write(&existing, Target::Pmk, false));
        assert!(any_would_write(
            &existing,
            &BTreeSet::from([Target::Frq, Target::Pmk]),
            false
        ));
        assert!(!any_would_write(
            &existing,
            &BTreeSet::from([Target::Frq]),
            false
        ));
    }

    #[test]
    fn a_clean_run_exits_0_even_with_warnings_and_skips() {
        let summary = RunSummary {
            considered: 3,
            written: 1,
            skipped: 2,
            failed: Vec::new(),
            warnings: vec![(PathBuf::from("bank/A2.wav"), "quiet warning".to_string())],
            cancelled: false,
        };
        assert_eq!(exit_code(&summary), 0);
    }

    #[test]
    fn any_failure_exits_1() {
        let summary = RunSummary {
            considered: 1,
            failed: vec![(PathBuf::from("bank/A2.wav"), "decode failed".to_string())],
            ..RunSummary::default()
        };
        assert_eq!(exit_code(&summary), 1);
    }

    #[test]
    fn cancellation_exits_130_even_with_failures() {
        let summary = RunSummary {
            considered: 2,
            written: 1,
            failed: vec![(PathBuf::from("bank/B2.wav"), "decode failed".to_string())],
            cancelled: true,
            ..RunSummary::default()
        };
        assert_eq!(exit_code(&summary), 130);
    }
}
