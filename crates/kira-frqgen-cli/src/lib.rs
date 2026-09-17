//! The `kira-frqgen` command-line interface (#12). The binary is a thin shell
//! around these modules: `args` owns the clap surface, `prompt` the single
//! interactive question's policy, and `report` the stderr lines; `main.rs`
//! wires them to the generation pipeline.

pub mod args;
pub mod prompt;
pub mod report;

use kira_frqgen::RunSummary;

/// #12's exit codes for a completed run: `0` clean — warnings, skips and
/// "nothing to do" included — `1` any per-file failure, `130` cancelled (a
/// partial summary was printed). Usage errors exit `2` through clap and fatal
/// input errors exit `1` through `main`.
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
