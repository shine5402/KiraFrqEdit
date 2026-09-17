//! The `kira-frqgen` binary (#12): plan, prompt, generate, report, exit.

use std::collections::BTreeSet;
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use clap::Parser;
use kira_frqgen::{CancelToken, GenerateOptions, Sharing, Target, WorldEstimator, generate, plan};
use kira_frqgen_cli::args::Cli;
use kira_frqgen_cli::report::{self, Reporter, Verbosity};
use kira_frqgen_cli::{exit_code, prompt};

fn main() -> ExitCode {
    ExitCode::from(run(Cli::parse()))
}

fn run(cli: Cli) -> u8 {
    let verbosity = if cli.quiet {
        Verbosity::Quiet
    } else if cli.verbose {
        Verbosity::Verbose
    } else {
        Verbosity::Normal
    };
    let targets = cli.targets();

    // Before the scan: Ctrl-C during it must still cancel the run (#12: 130
    // with a partial summary), not surface the platform's signal exit.
    let cancel: CancelToken = Arc::new(AtomicBool::new(false));
    let handler_token = Arc::clone(&cancel);
    if let Err(error) = ctrlc::set_handler(move || {
        handler_token.store(true, Ordering::SeqCst);
    }) && verbosity != Verbosity::Quiet
    {
        eprintln!("warning: cannot install the Ctrl-C handler: {error}");
    }

    let mut config_warnings = Vec::new();
    let mut opts = GenerateOptions {
        root: cli.path.clone(),
        targets: targets.clone(),
        overwrite: cli.overwrite,
        f0: cli.f0_config(),
        jobs: cli.jobs,
        sharing: sharing(&cli, &targets, &mut config_warnings),
        // Whether caches are deleted is re-resolved below, against the plan.
        delete_llsm: true,
    };

    let started = Instant::now();
    let plan = match plan(&opts) {
        Ok(plan) => plan,
        Err(error) => {
            eprintln!("error: {error}");
            return 1;
        }
    };
    if verbosity != Verbosity::Quiet {
        for warning in &config_warnings {
            eprintln!("warning: {warning}");
        }
    }

    opts.delete_llsm = if cancel.load(Ordering::Relaxed) {
        true
    } else {
        let planned = prompt::f0_write_planned(&plan, &targets, cli.overwrite);
        prompt::resolve_llsm(cli.explicit_llsm(), cli.yes, cli.dry_run, planned, || {
            if prompt::interactive() {
                prompt::ask_llsm()
            } else {
                true
            }
        })
    };

    if cli.dry_run {
        if verbosity != Verbosity::Quiet {
            for file in &plan.files {
                eprintln!("{}", report::plan_line(file, &targets, cli.overwrite));
            }
            eprintln!(
                "{}",
                report::plan_summary_line(&plan, &targets, cli.overwrite, started.elapsed())
            );
        }
        return if cancel.load(Ordering::Relaxed) {
            130
        } else {
            0
        };
    }

    let estimator = WorldEstimator::new(opts.f0);
    let reporter = Reporter::new(verbosity);
    match generate(&opts, &estimator, &reporter, &cancel) {
        Ok(summary) => exit_code(&summary),
        Err(error) => {
            eprintln!("error: {error}");
            1
        }
    }
}

/// `--ensure-japanese-codepage` (#10/#12): mrq-only, a silent no-op on code
/// page 932, and a warned no-op on platforms without an 8-bit layer.
fn sharing(cli: &Cli, targets: &BTreeSet<Target>, warnings: &mut Vec<String>) -> Option<Sharing> {
    if !cli.ensure_japanese_codepage {
        return None;
    }
    if !targets.contains(&Target::Mrq) {
        warnings.push(
            "--ensure-japanese-codepage has no effect: mrq is not among the selected formats"
                .to_string(),
        );
        return None;
    }
    if !cfg!(windows) {
        warnings.push(
            "--ensure-japanese-codepage is a no-op on this platform: it has no 8-bit code page"
                .to_string(),
        );
        return None;
    }
    let sharing = Sharing::native();
    if sharing.code_page() == 932 {
        return None;
    }
    Some(sharing)
}
