//! The `kirafrqgen-cli` binary (#12): plan, prompt, generate, report, exit.

use std::collections::BTreeSet;
use std::io::IsTerminal;
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use clap::Parser;
use kirafrqgen_cli::args::Cli;
use kirafrqgen_cli::report::{self, Reporter, Verbosity};
use kirafrqgen_cli::{exit_code, prompt};
use kirafrqgen_core::{
    CancelToken, Estimator, GenerateOptions, Sharing, Target, build_estimator, generate, plan,
};

fn main() -> ExitCode {
    ExitCode::from(run(Cli::parse()))
}

fn run(cli: Cli) -> u8 {
    if cli.license || cli.license_full {
        // Style for a terminal; plain text for a pipe or when NO_COLOR is set.
        let style = if std::io::stdout().is_terminal() && std::env::var_os("NO_COLOR").is_none() {
            kirafrq_credits::Style::Ansi
        } else {
            kirafrq_credits::Style::Plain
        };
        let credits =
            kirafrq_credits::render(kirafrq_credits::for_display(cli.license_full), style);
        print!("{credits}");
        return 0;
    }
    let path = cli
        .path
        .clone()
        .expect("clap requires PATH unless a license flag was given");
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
        root: path,
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
        let mrq_planned = prompt::mrq_write_planned(&plan, &targets, cli.overwrite);
        prompt::resolve_llsm(
            cli.explicit_llsm(),
            cli.yes,
            cli.dry_run,
            mrq_planned,
            || {
                if prompt::interactive() {
                    prompt::ask_llsm()
                } else {
                    true
                }
            },
        )
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

    // The estimator factory resolves the ML model up front (#49): a missing
    // model is a single early error, before any file is touched. The quirks
    // warning fires first, so an explicit switch is always acknowledged.
    if let Some(warning) = world_quirks_warning(&cli, verbosity) {
        eprintln!("warning: {warning}");
    }
    let estimator = match build_estimator(&opts.f0, opts.jobs) {
        Ok(estimator) => estimator,
        Err(error) => {
            eprintln!("error: {error}");
            return 1;
        }
    };
    let reporter = Reporter::with_progress(verbosity, plan.files.len(), cli.progress);
    match generate(&opts, estimator.as_ref(), &reporter, &cancel) {
        Ok(summary) => exit_code(&summary),
        Err(error) => {
            eprintln!("error: {error}");
            1
        }
    }
}

/// #49: an explicit WORLD-quirks switch alongside a non-WORLD estimator is a
/// warned no-op; the run continues.
fn world_quirks_warning(cli: &Cli, verbosity: Verbosity) -> Option<String> {
    if !cli.no_world_quirks || verbosity == Verbosity::Quiet {
        return None;
    }
    if cli.estimator() == Estimator::Rmvpe {
        return Some(
            "--no-world-quirks has no effect: it applies to the WORLD estimators only".to_string(),
        );
    }
    None
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
