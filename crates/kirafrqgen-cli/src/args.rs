//! The `kirafrqgen-cli` flags (#12): one required PATH, no subcommand.

use std::collections::BTreeSet;
use std::path::PathBuf;

use clap::{Parser, ValueEnum};
use kirafrqgen_core::{Estimator, F0Config, Target};

/// Bulk-generate frq tables for your UTAU voicebank.
#[derive(Debug, Parser)]
#[command(
    name = "kirafrqgen-cli",
    version,
    about = "Bulk-generate frq tables for your UTAU voicebank."
)]
pub struct Cli {
    /// A voicebank folder (recursive scan) or a single .wav file.
    #[arg(value_name = "PATH")]
    pub path: PathBuf,

    /// Table formats to write: frq, pmk, mrq (comma-separated, repeatable).
    #[arg(long, value_delimiter = ',', value_name = "FORMAT")]
    pub format: Vec<FormatArg>,

    /// f0 estimator for every wav. Defaults to rmvpe when a model is present
    /// in an ML-enabled build, else dio.
    #[arg(long, value_enum)]
    pub estimator: Option<EstimatorArg>,

    /// Disable the tuned WORLD path (the energy voicing gate); WORLD only.
    #[arg(long)]
    pub no_world_quirks: bool,

    /// Regenerate tables that already exist.
    #[arg(long)]
    pub overwrite: bool,

    /// Worker threads: `auto` (all cores) or a count.
    #[arg(short, long, default_value = "auto", value_name = "auto|N", value_parser = parse_jobs)]
    pub jobs: usize,

    /// Scan and report what would happen; no analysis, writes or prompts.
    #[arg(long)]
    pub dry_run: bool,

    /// Failures only; no summary.
    #[arg(short, long, conflicts_with = "verbose")]
    pub quiet: bool,

    /// One line per file: outcome, targets and reasons.
    #[arg(short, long)]
    pub verbose: bool,

    /// Delete `.llsm` caches after mrq writes (default).
    #[arg(long, conflicts_with = "no_delete_llsm")]
    pub delete_llsm: bool,

    /// Keep `.llsm` caches.
    #[arg(long)]
    pub no_delete_llsm: bool,

    /// mrq: also key entries by the Japanese-side filename.
    #[arg(long)]
    pub ensure_japanese_codepage: bool,

    /// Approve prompts without asking.
    #[arg(short = 'y', long)]
    pub yes: bool,

    /// Show live run progress on stderr (TTY only).
    #[arg(long)]
    pub progress: bool,
}

impl Cli {
    /// The selected formats (#12).
    pub fn targets(&self) -> BTreeSet<Target> {
        let mut targets: BTreeSet<Target> = self.format.iter().copied().map(Target::from).collect();
        if targets.is_empty() {
            targets.insert(Target::Frq);
        }
        targets
    }

    /// The standing f0 defaults (#7/#8) with the resolved estimator and the
    /// WORLD-quirks toggle (#49).
    pub fn f0_config(&self) -> F0Config {
        F0Config {
            estimator: self.estimator(),
            world_quirks: !self.no_world_quirks,
            ..F0Config::default()
        }
    }

    /// The capability-aware default (#49): RMVPE when the build has ML and a
    /// model file resolves, else DIO. An explicit `--estimator` always wins.
    pub fn estimator(&self) -> Estimator {
        if let Some(explicit) = self.estimator {
            return explicit.into();
        }
        let default = F0Config::default();
        if kirafrqgen_core::ml_available(&default) {
            Estimator::Rmvpe
        } else {
            Estimator::Dio
        }
    }

    /// The explicit `.llsm` flags as a value; `None` when neither was passed,
    /// so the prompt may decide (#12).
    pub fn explicit_llsm(&self) -> Option<bool> {
        if self.delete_llsm {
            Some(true)
        } else if self.no_delete_llsm {
            Some(false)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum FormatArg {
    Frq,
    Pmk,
    Mrq,
}

impl From<FormatArg> for Target {
    fn from(value: FormatArg) -> Self {
        match value {
            FormatArg::Frq => Target::Frq,
            FormatArg::Pmk => Target::Pmk,
            FormatArg::Mrq => Target::Mrq,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum EstimatorArg {
    /// Fast, but may struggle on less-than-ideal recordings. A traditional
    /// DSP-based algorithm from WORLD.
    Dio,
    /// Robust, but slow. A traditional DSP-based algorithm from WORLD.
    Harvest,
    /// Fast and reliable ML based estimator. Requires model to be present.
    Rmvpe,
}

impl From<EstimatorArg> for Estimator {
    fn from(value: EstimatorArg) -> Self {
        match value {
            EstimatorArg::Harvest => Estimator::Harvest,
            EstimatorArg::Dio => Estimator::Dio,
            EstimatorArg::Rmvpe => Estimator::Rmvpe,
        }
    }
}

/// `--jobs`: `auto` maps to the pipeline's "all cores" marker, `0`.
fn parse_jobs(value: &str) -> Result<usize, String> {
    if value == "auto" {
        return Ok(0);
    }
    match value.parse::<usize>() {
        Ok(0) => Err("a worker count of 0 is invalid; use `auto` for all cores".to_string()),
        Ok(count) => Ok(count),
        Err(_) => Err(format!("expected `auto` or a worker count, got `{value}`")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<Cli, clap::Error> {
        Cli::try_parse_from(std::iter::once("kirafrqgen-cli").chain(args.iter().copied()))
    }

    fn parse_ok(args: &[&str]) -> Cli {
        parse(args).unwrap_or_else(|error| panic!("expected {args:?} to parse: {error}"))
    }

    #[test]
    fn frq_is_the_default_format() {
        let cli = parse_ok(&["bank"]);
        assert_eq!(cli.targets(), BTreeSet::from([Target::Frq]));
    }

    #[test]
    fn format_parses_comma_lists_and_repeats() {
        let cli = parse_ok(&["bank", "--format", "pmk,mrq", "--format", "frq"]);
        assert_eq!(
            cli.targets(),
            BTreeSet::from([Target::Frq, Target::Pmk, Target::Mrq])
        );
    }

    #[test]
    fn unknown_or_uppercase_formats_are_usage_errors() {
        for args in [
            &["bank", "--format", "flac"][..],
            &["bank", "--format", "FRQ"][..],
            &["bank", "--format", "frq,"][..],
        ] {
            let error = parse(args).unwrap_err();
            assert_eq!(error.exit_code(), 2, "{args:?}");
        }
    }

    #[test]
    fn jobs_accepts_auto_and_counts_but_rejects_zero() {
        assert_eq!(parse_ok(&["bank"]).jobs, 0, "auto is the default");
        assert_eq!(parse_ok(&["bank", "-j", "auto"]).jobs, 0);
        assert_eq!(parse_ok(&["bank", "-j", "3"]).jobs, 3);
        assert_eq!(parse_ok(&["bank", "--jobs", "1"]).jobs, 1);

        for args in [
            &["bank", "-j", "0"][..],
            &["bank", "-j", "-1"][..],
            &["bank", "-j", "many"][..],
        ] {
            let error = parse(args).unwrap_err();
            assert_eq!(error.exit_code(), 2, "{args:?}");
        }
    }

    #[test]
    fn estimator_is_unset_by_default_and_accepts_every_name() {
        assert_eq!(parse_ok(&["bank"]).estimator, None);
        assert_eq!(
            parse_ok(&["bank", "--estimator", "dio"]).estimator,
            Some(EstimatorArg::Dio)
        );
        assert_eq!(
            parse_ok(&["bank", "--estimator", "harvest"]).estimator,
            Some(EstimatorArg::Harvest)
        );
        assert_eq!(
            parse_ok(&["bank", "--estimator", "rmvpe"]).estimator,
            Some(EstimatorArg::Rmvpe)
        );
        let error = parse(&["bank", "--estimator", "swipe"]).unwrap_err();
        assert_eq!(error.exit_code(), 2);
    }

    #[test]
    fn an_explicit_estimator_always_wins_over_the_default() {
        // The capability-aware default (#49) resolves to DIO here (no model
        // in the test environment); an explicit flag must beat it.
        assert_eq!(
            parse_ok(&["bank", "--estimator", "rmvpe"]).estimator(),
            Estimator::Rmvpe
        );
        assert_eq!(
            parse_ok(&["bank", "--estimator", "harvest"]).estimator(),
            Estimator::Harvest
        );
    }

    #[test]
    fn world_quirks_default_on_and_no_world_quirks_turns_them_off() {
        assert!(parse_ok(&["bank"]).f0_config().world_quirks);
        assert!(
            !parse_ok(&["bank", "--no-world-quirks"])
                .f0_config()
                .world_quirks
        );
    }

    #[test]
    fn quiet_and_verbose_are_mutually_exclusive() {
        let error = parse(&["bank", "-q", "-v"]).unwrap_err();
        assert_eq!(error.exit_code(), 2);
        let error = parse(&["bank", "--quiet", "--verbose"]).unwrap_err();
        assert_eq!(error.exit_code(), 2);
    }

    #[test]
    fn llsm_flags_are_mutually_exclusive_and_resolve_to_a_value() {
        assert_eq!(parse_ok(&["bank"]).explicit_llsm(), None);
        assert_eq!(
            parse_ok(&["bank", "--delete-llsm"]).explicit_llsm(),
            Some(true)
        );
        assert_eq!(
            parse_ok(&["bank", "--no-delete-llsm"]).explicit_llsm(),
            Some(false)
        );
        let error = parse(&["bank", "--delete-llsm", "--no-delete-llsm"]).unwrap_err();
        assert_eq!(error.exit_code(), 2);
    }

    #[test]
    fn f0_config_uses_the_standing_defaults() {
        let cli = parse_ok(&["bank", "--estimator", "dio"]);
        let f0 = cli.f0_config();
        assert_eq!(f0.estimator, Estimator::Dio);
        assert_eq!(f0.floor_hz, 71.0);
        assert_eq!(f0.ceiling_hz, 800.0);
        assert!(f0.stone_mask);
    }

    #[test]
    fn the_path_is_required_and_extra_positionals_are_rejected() {
        assert_eq!(parse(&[]).unwrap_err().exit_code(), 2);
        assert_eq!(parse(&["a", "b"]).unwrap_err().exit_code(), 2);
        assert_eq!(parse(&["--nope", "a"]).unwrap_err().exit_code(), 2);
    }

    #[test]
    fn a_double_dash_lets_a_path_that_looks_like_a_flag_through() {
        let cli = parse_ok(&["--", "--weird.wav"]);
        assert_eq!(cli.path, PathBuf::from("--weird.wav"));
    }

    #[test]
    fn dry_run_and_yes_are_flags() {
        let cli = parse_ok(&["bank", "--dry-run", "-y"]);
        assert!(cli.dry_run);
        assert!(cli.yes);
        assert!(!cli.ensure_japanese_codepage);
    }

    #[test]
    fn progress_is_opt_in_and_orthogonal_to_verbosity() {
        assert!(!parse_ok(&["bank"]).progress);
        assert!(parse_ok(&["bank", "--progress"]).progress);
        // No short form, and it combines with either report mode.
        assert!(parse_ok(&["bank", "--progress", "-v"]).progress);
        assert!(parse_ok(&["bank", "--progress", "-q"]).progress);
        let error = parse(&["bank", "--progress=x"]).unwrap_err();
        assert_eq!(error.exit_code(), 2);
    }
}
