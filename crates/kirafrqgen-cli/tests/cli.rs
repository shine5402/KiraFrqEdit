//! Black-box tests for the `kirafrqgen-cli` binary (#12): flags, report, exit
//! codes, sidecar rules and prompt behavior, over scratch folders.
//!
//! Runs that need a real table use a short harmonic tone; dry-run and
//! all-existing cases never decode, so they use placeholder bytes. The
//! interactive ask loop itself is unit-tested in `prompt.rs` — a spawned
//! process has no TTY, so here only the non-TTY default is observable.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};

use hound::{SampleFormat, WavSpec, WavWriter};
use kirafrq_formats::{frq, mrq};

const BIN: &str = env!("CARGO_BIN_EXE_kirafrqgen-cli");

// --- scratch folder ---------------------------------------------------------

struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "kirafrqgen-cli-{tag}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }

    fn dir(&self, name: &str) -> PathBuf {
        let path = self.path(name);
        fs::create_dir_all(&path).unwrap();
        path
    }

    fn write(&self, name: &str, bytes: &[u8]) -> PathBuf {
        let path = self.path(name);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(&path, bytes).unwrap();
        path
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

// --- running the binary -----------------------------------------------------

struct Run {
    code: i32,
    stdout: String,
    stderr: String,
}

impl Run {
    fn stdout_is_empty(&self) -> bool {
        self.stdout.is_empty()
    }
}

fn cli(dir: &Path, args: &[&str]) -> Run {
    cli_with_env(dir, args, &[])
}

/// Run the binary with extra environment variables, for the ML model lookup.
fn cli_with_env(dir: &Path, args: &[&str], env: &[(&str, &str)]) -> Run {
    let mut command = Command::new(BIN);
    command.args(args).current_dir(dir).stdin(Stdio::null());
    for (key, value) in env {
        command.env(key, value);
    }
    let output: Output = command.output().expect("kirafrqgen-cli runs");
    Run {
        code: output.status.code().expect("no signal termination"),
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    }
}

fn run_ok(dir: &Path, args: &[&str]) -> Run {
    let run = cli(dir, args);
    assert_eq!(run.code, 0, "args {args:?}: {}", run.stderr);
    assert!(
        run.stdout_is_empty(),
        "stdout must stay empty: {:?}",
        run.stdout
    );
    run
}

// --- wav fixtures -----------------------------------------------------------

fn write_tone(scratch: &Scratch, name: &str) {
    let path = scratch.path(name);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    let spec = WavSpec {
        channels: 1,
        sample_rate: 44_100,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut writer = WavWriter::create(&path, spec).unwrap();
    for sample in tone(0.3) {
        writer.write_sample(sample).unwrap();
    }
    writer.finalize().unwrap();
}

/// A harmonic tone: WORLD finds f0 in it, and any values are fine for the
/// sidecar rules the CLI tests check.
fn tone(seconds: f64) -> Vec<i16> {
    const RATE: f64 = 44_100.0;
    let count = (RATE * seconds) as usize;
    (0..count)
        .map(|n| {
            let t = n as f64 / RATE;
            let sample: f64 = (1..=10)
                .map(|harmonic| {
                    (0.4 / harmonic as f64)
                        * (2.0 * std::f64::consts::PI * 220.0 * harmonic as f64 * t).sin()
                })
                .sum();
            (sample * 0.8 * i16::MAX as f64).clamp(i16::MIN as f64, i16::MAX as f64) as i16
        })
        .collect()
}

fn utf16(name: &str) -> Vec<u16> {
    name.encode_utf16().collect()
}

// --- usage errors and help --------------------------------------------------

#[test]
fn usage_errors_exit_2_with_an_empty_stdout() {
    let scratch = Scratch::new("usage");
    let cases: &[&[&str]] = &[
        &[],
        &["one", "two"],
        &["bank", "--nope"],
        &["bank", "--format", "flac"],
        &["bank", "--format", "FRQ"],
        &["bank", "--format", "frq,"],
        &["bank", "--jobs", "0"],
        &["bank", "--jobs", "many"],
        &["bank", "--estimator", "swipe"],
        &["bank", "-q", "-v"],
        &["bank", "--delete-llsm", "--no-delete-llsm"],
    ];
    for args in cases {
        let run = cli(&scratch.0, args);
        assert_eq!(run.code, 2, "args {args:?}: {}", run.stderr);
        assert!(
            run.stdout_is_empty(),
            "args {args:?}: stdout {:?}",
            run.stdout
        );
    }
}

#[test]
fn help_and_version_print_to_stdout_and_exit_0() {
    let scratch = Scratch::new("help");
    let help = cli(&scratch.0, &["--help"]);
    assert_eq!(help.code, 0, "{}", help.stderr);
    assert!(
        help.stdout
            .contains("Bulk-generate frq tables for your UTAU voicebank."),
        "{}",
        help.stdout
    );

    let version = cli(&scratch.0, &["-V"]);
    assert_eq!(version.code, 0, "{}", version.stderr);
    assert_eq!(
        version.stdout.trim(),
        format!("kirafrqgen-cli {}", env!("CARGO_PKG_VERSION"))
    );
}

// --- fatal input errors -----------------------------------------------------

#[test]
fn fatal_input_errors_exit_1() {
    let scratch = Scratch::new("fatal");
    scratch.dir("empty");
    scratch.write("notes.txt", b"not a wav");

    for args in [&["missing"][..], &["empty"][..], &["notes.txt"][..]] {
        let run = cli(&scratch.0, args);
        assert_eq!(run.code, 1, "args {args:?}: {}", run.stderr);
        assert!(
            run.stderr.contains("error:"),
            "args {args:?}: {}",
            run.stderr
        );
        assert!(
            run.stdout_is_empty(),
            "args {args:?}: stdout {:?}",
            run.stdout
        );
    }
}

// --- dry run ----------------------------------------------------------------

#[test]
fn dry_run_reports_would_write_and_touches_nothing() {
    let scratch = Scratch::new("dry-run");
    scratch.write("bank/A2.wav", b"placeholder, never decoded");

    let run = run_ok(&scratch.0, &["bank", "--dry-run"]);
    assert!(run.stderr.contains("plan: bank"), "{}", run.stderr);
    assert!(run.stderr.contains("would write frq"), "{}", run.stderr);
    assert!(run.stderr.contains("considered 1"), "{}", run.stderr);
    assert_eq!(run.stderr.lines().count(), 2, "{}", run.stderr);
    assert!(!scratch.path("bank/A2_wav.frq").exists());
}

#[test]
fn dry_run_counts_both_frq_spellings_and_the_overwrite_policy() {
    let scratch = Scratch::new("dry-run-existing");
    scratch.write("bank/A2.wav", b"x");
    scratch.write("bank/A2_wav.frq", b"canonical");
    scratch.write("bank/B2.wav", b"x");
    scratch.write("bank/B2.wav.frq", b"alternate");

    let run = run_ok(&scratch.0, &["bank", "--dry-run"]);
    assert!(
        run.stderr
            .contains("considered 2, would write 0, would skip 2"),
        "{}",
        run.stderr
    );
    assert!(
        run.stderr.matches("would skip frq").count() == 2,
        "{}",
        run.stderr
    );

    let run = run_ok(&scratch.0, &["bank", "--dry-run", "--overwrite"]);
    assert!(
        run.stderr
            .contains("considered 2, would write 2, would skip 0"),
        "{}",
        run.stderr
    );
}

#[test]
fn dry_run_reports_paths_as_given() {
    let scratch = Scratch::new("dry-run-relative");
    scratch.write("bank/A2.wav", b"x");

    let run = run_ok(&scratch.0, &["bank", "--dry-run"]);
    assert!(run.stderr.contains("bank"), "{}", run.stderr);
    let absolute = scratch.0.display().to_string();
    assert!(
        !run.stderr.contains(&absolute),
        "expected relative paths, got: {}",
        run.stderr
    );
}

// --- writes -----------------------------------------------------------------

#[test]
fn a_run_writes_frq_and_the_next_run_skips_it() {
    let scratch = Scratch::new("write");
    write_tone(&scratch, "bank/A2.wav");

    let first = run_ok(&scratch.0, &["bank"]);
    assert!(first.stderr.contains("written 1"), "{}", first.stderr);
    let table = scratch.path("bank/A2_wav.frq");
    assert!(table.exists());
    let bytes = fs::read(&table).unwrap();
    frq::read(&table).expect("the written table is a valid frq");

    let second = run_ok(&scratch.0, &["bank"]);
    assert!(second.stderr.contains("skipped 1"), "{}", second.stderr);
    assert_eq!(second.stderr.lines().count(), 1, "{}", second.stderr);
    assert_eq!(fs::read(&table).unwrap(), bytes, "a skip never rewrites");
}

#[test]
fn overwrite_replaces_the_alternate_frq_spelling() {
    let scratch = Scratch::new("alt-frq");
    write_tone(&scratch, "bank/A2.wav");
    scratch.write("bank/A2.wav.frq", b"the alternate spelling");

    let kept = run_ok(&scratch.0, &["bank"]);
    assert!(kept.stderr.contains("skipped 1"), "{}", kept.stderr);
    assert!(!scratch.path("bank/A2_wav.frq").exists());
    assert!(scratch.path("bank/A2.wav.frq").exists());

    let replaced = run_ok(&scratch.0, &["bank", "--overwrite"]);
    assert!(replaced.stderr.contains("written 1"), "{}", replaced.stderr);
    assert!(
        !scratch.path("bank/A2.wav.frq").exists(),
        "the alternate is deleted"
    );
    frq::read(&scratch.path("bank/A2_wav.frq")).expect("the canonical table is valid");
}

#[test]
fn pmk_and_mrq_are_opt_in_sidecars() {
    let scratch = Scratch::new("sidecars");
    write_tone(&scratch, "bank/A2.wav");

    let pmk = run_ok(
        &scratch.0,
        &["bank", "--format", "pmk", "--estimator", "dio", "-v"],
    );
    assert!(pmk.stderr.contains("wrote pmk"), "{}", pmk.stderr);
    assert!(scratch.path("bank/A2_wav.pmk").exists());
    assert!(!scratch.path("bank/A2_wav.frq").exists());

    let mrq = run_ok(&scratch.0, &["bank", "--format", "mrq", "-v"]);
    assert!(mrq.stderr.contains("wrote mrq"), "{}", mrq.stderr);
    let desc = mrq::Desc::read(&mrq::desc_path(&scratch.path("bank")))
        .unwrap()
        .expect("desc.mrq exists");
    assert!(desc.has(&utf16("A2.wav")), "the wav has an entry");
    assert!(!scratch.path("bank/A2_wav.frq").exists());
}

#[test]
fn a_failed_file_exits_1_and_never_stops_the_run() {
    let scratch = Scratch::new("failure");
    write_tone(&scratch, "bank/good.wav");
    scratch.write("bank/bad.wav", b"definitely not a wav");

    let run = cli(&scratch.0, &["bank"]);
    assert_eq!(run.code, 1, "{}", run.stderr);
    assert!(run.stderr.contains("bad.wav"), "{}", run.stderr);
    assert!(run.stderr.contains("error:"), "{}", run.stderr);
    assert!(run.stderr.contains("failed 1"), "{}", run.stderr);
    assert!(
        scratch.path("bank/good_wav.frq").exists(),
        "the rest still runs"
    );
    assert!(run.stdout_is_empty());
}

// --- llsm -------------------------------------------------------------------

#[test]
fn an_frq_only_run_leaves_llsm_caches_alone() {
    let scratch = Scratch::new("llsm-frq");
    write_tone(&scratch, "bank/A2.wav");
    scratch.write("bank/A2.wav.llsm", b"cache");

    let run = run_ok(&scratch.0, &["bank", "--overwrite"]);
    assert!(run.stderr.contains("written 1"), "{}", run.stderr);
    assert!(
        !run.stderr.contains("[Y/n]"),
        "frq-only never prompts: {}",
        run.stderr
    );
    assert!(
        scratch.path("bank/A2.wav.llsm").exists(),
        "frq is not moresampler's f0 source"
    );
}

#[test]
fn mrq_writes_delete_llsm_by_default_without_a_prompt_outside_a_tty() {
    let scratch = Scratch::new("llsm-mrq");
    write_tone(&scratch, "bank/A2.wav");
    scratch.write("bank/A2.wav.llsm", b"cache");

    let run = run_ok(&scratch.0, &["bank", "--format", "mrq"]);
    assert!(run.stderr.contains("written 1"), "{}", run.stderr);
    assert!(
        !run.stderr.contains("[Y/n]"),
        "no prompt without a TTY: {}",
        run.stderr
    );
    assert!(!scratch.path("bank/A2.wav.llsm").exists());
}

#[test]
fn no_delete_llsm_keeps_the_cache() {
    let scratch = Scratch::new("llsm-keep");
    write_tone(&scratch, "bank/A2.wav");
    scratch.write("bank/A2.wav.llsm", b"cache");

    let run = run_ok(&scratch.0, &["bank", "--format", "mrq", "--no-delete-llsm"]);
    assert!(run.stderr.contains("written 1"), "{}", run.stderr);
    assert!(scratch.path("bank/A2.wav.llsm").exists());
}

// --- report modes -----------------------------------------------------------

#[test]
fn quiet_keeps_failures_only() {
    let scratch = Scratch::new("quiet");
    scratch.write("bank/bad.wav", b"definitely not a wav");
    scratch.write("bank2/B2.wav", b"placeholder");
    scratch.write("bank2/B2_wav.frq", b"existing table");

    let failed = cli(&scratch.0, &["bank", "-q"]);
    assert_eq!(failed.code, 1, "{}", failed.stderr);
    assert!(failed.stderr.contains("error:"), "{}", failed.stderr);
    assert!(
        !failed.stderr.contains("considered"),
        "no summary: {}",
        failed.stderr
    );
    assert!(failed.stdout_is_empty());

    let clean = run_ok(&scratch.0, &["bank2", "-q"]);
    assert!(
        clean.stderr.is_empty(),
        "quiet clean run is silent: {}",
        clean.stderr
    );

    let dry = run_ok(&scratch.0, &["bank2", "-q", "--dry-run"]);
    assert!(
        dry.stderr.is_empty(),
        "quiet dry run is silent: {}",
        dry.stderr
    );
}

#[test]
fn verbose_lists_every_file_with_targets_and_reasons() {
    let scratch = Scratch::new("verbose");
    write_tone(&scratch, "bank/A2.wav");
    scratch.write("bank/B2.wav", b"placeholder");
    scratch.write("bank/B2_wav.frq", b"existing table");

    let run = run_ok(&scratch.0, &["bank", "-v"]);
    assert!(run.stderr.contains("A2.wav: wrote frq"), "{}", run.stderr);
    assert!(
        run.stderr.contains("B2.wav: skipped frq (exists)"),
        "{}",
        run.stderr
    );
    assert!(run.stderr.contains("considered 2"), "{}", run.stderr);
    assert_eq!(run.stderr.lines().count(), 3, "{}", run.stderr);
}

#[test]
fn progress_is_silent_without_a_tty_and_keeps_stdout_empty() {
    // The spawned process has piped stderr (no TTY), so the flag must not
    // add output: logs read exactly as without it.
    let scratch = Scratch::new("progress-piped");
    write_tone(&scratch, "bank/A2.wav");

    let plain = run_ok(&scratch.0, &["bank", "--overwrite", "-v"]);
    let flagged = run_ok(&scratch.0, &["bank", "--overwrite", "-v", "--progress"]);
    assert!(!flagged.stderr.contains('\r'), "{}", flagged.stderr);
    assert!(
        !flagged.stderr.contains("files finished"),
        "{}",
        flagged.stderr
    );
    // Same completion lines; only the elapsed time may differ.
    fn strip_elapsed(stderr: &str) -> Vec<&str> {
        stderr
            .lines()
            .map(|line| line.split_once(", elapsed").map_or(line, |(head, _)| head))
            .collect()
    }
    assert_eq!(
        strip_elapsed(&flagged.stderr),
        strip_elapsed(&plain.stderr),
        "no extra output"
    );
    assert!(flagged.stdout_is_empty());

    // `-q` and `--dry-run` suppress the line too (here trivially: there is
    // none without a TTY), while stdout stays empty.
    let quiet = run_ok(&scratch.0, &["bank", "--progress", "-q"]);
    assert!(!quiet.stderr.contains('\r'), "{}", quiet.stderr);
    let dry = run_ok(&scratch.0, &["bank", "--progress", "--dry-run"]);
    assert!(!dry.stderr.contains('\r'), "{}", dry.stderr);
    assert!(!dry.stderr.contains("files finished"), "{}", dry.stderr);
    assert!(dry.stdout_is_empty());
}

// --- ML estimator (#49) -----------------------------------------------------

/// The committed tiny ONNX fixture, copied into `models/rmvpe.onnx`; returns
/// the directory to point `KIRAFRQ_ML_DIR` at.
fn fixture_model_dir(scratch: &Scratch) -> String {
    let model_dir = scratch.dir("models");
    fs::copy(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("kirafrq-ml-provider")
            .join("tests")
            .join("fixtures")
            .join("rmvpe_tiny.onnx"),
        model_dir.join("rmvpe.onnx"),
    )
    .unwrap();
    model_dir.to_str().unwrap().to_string()
}

/// The CLI is spawned without a model file next to it, so modern `rmvpe`
/// must fail early with the actionable hint rather than touching any file;
/// the compat build has no `rmvpe` value at all.
#[test]
fn rmvpe_without_a_model_errors_with_the_download_hint() {
    let scratch = Scratch::new("rmvpe-missing");
    scratch.write("bank/A2.wav", b"placeholder");

    let run = cli(&scratch.0, &["bank", "--estimator", "rmvpe"]);
    assert!(run.stdout_is_empty());
    if cfg!(feature = "ml") {
        assert_eq!(run.code, 1, "{}", run.stderr);
        assert!(run.stderr.contains("error:"), "{}", run.stderr);
        assert!(run.stderr.contains("rmvpe.onnx"), "{}", run.stderr);
        assert!(run.stderr.contains("KIRAFRQ_ML_DIR"), "{}", run.stderr);
        assert!(
            !scratch.path("bank/A2_wav.frq").exists(),
            "the run fails before any write"
        );
    } else {
        assert_eq!(run.code, 2, "a usage error: {}", run.stderr);
        assert!(run.stderr.contains("invalid value"), "{}", run.stderr);
    }
}

#[test]
fn no_world_quirks_with_rmvpe_warns_and_continues() {
    if !cfg!(feature = "ml") {
        // The compat build has no `rmvpe` value, so there is nothing to warn
        // about; the usage error is covered above.
        return;
    }
    let scratch = Scratch::new("rmvpe-quirks");
    scratch.write("bank/A2.wav", b"placeholder");

    // No model in the test environment: the run fails at the factory, but the
    // warning is emitted first and the failure is the model error, not the
    // flag.
    let run = cli(
        &scratch.0,
        &["bank", "--estimator", "rmvpe", "--no-world-quirks"],
    );
    assert_eq!(run.code, 1, "{}", run.stderr);
    assert!(
        run.stderr.contains("--no-world-quirks has no effect"),
        "{}",
        run.stderr
    );
}

#[test]
fn no_world_quirks_with_a_world_estimator_is_silent() {
    let scratch = Scratch::new("world-quirks");
    write_tone(&scratch, "bank/A2.wav");

    let run = run_ok(
        &scratch.0,
        &["bank", "--estimator", "dio", "--no-world-quirks"],
    );
    assert!(
        !run.stderr.contains("no effect"),
        "a WORLD estimator accepts the flag: {}",
        run.stderr
    );
    assert!(scratch.path("bank/A2_wav.frq").exists());
}

#[test]
fn help_lists_every_estimator_name_this_build_has() {
    let scratch = Scratch::new("estimator-help");
    let run = cli(&scratch.0, &["--help"]);
    assert_eq!(run.code, 0, "{}", run.stderr);
    for name in ["dio", "harvest"] {
        assert!(run.stdout.contains(name), "{name} missing: {}", run.stdout);
    }
    // #48/#69: the compat build has no ML options, so it must not advertise
    // any.
    for name in ["rmvpe", "swiftf0"] {
        assert_eq!(
            run.stdout.contains(name),
            cfg!(feature = "ml"),
            "{name}: {}",
            run.stdout
        );
    }
    assert!(run.stderr.is_empty(), "help goes to stdout: {}", run.stderr);
}

/// The committed tiny ONNX fixture has RMVPE's I/O contract; with it in
/// `KIRAFRQ_ML_DIR` the CLI must run the ML path end to end (modern build
/// only) and write a table that keeps the #8 conventions.
#[test]
fn rmvpe_runs_end_to_end_with_the_committed_fixture() {
    if !cfg!(feature = "ml") {
        return;
    }
    let scratch = Scratch::new("rmvpe-fixture");
    write_tone(&scratch, "bank/A2.wav");
    let model_dir = fixture_model_dir(&scratch);

    let run = cli_with_env(
        &scratch.0,
        &["bank", "--estimator", "rmvpe", "-v"],
        &[("KIRAFRQ_ML_DIR", &model_dir)],
    );
    assert_eq!(run.code, 0, "{}", run.stderr);
    assert!(run.stderr.contains("wrote frq"), "{}", run.stderr);

    let table = frq::read(&scratch.path("bank/A2_wav.frq")).unwrap();
    assert_eq!(table.sample_rate, 44_100);
    assert_eq!(table.hop_samples, 256);
    for &value in &table.f0_hz {
        assert!(value.is_finite(), "non-finite f0: {value}");
        assert!((71.0..=800.0).contains(&value) || value == 0.0, "{value}");
    }
    assert_eq!(*table.f0_hz.last().unwrap(), 0.0, "the trailing rule");
}

/// #69: SwiftF0 bundles its MIT model, so `--estimator swiftf0` must run end
/// to end with nothing on disk (modern build only).
#[test]
fn swiftf0_runs_end_to_end_with_the_bundled_model() {
    if !cfg!(feature = "ml") {
        return;
    }
    let scratch = Scratch::new("swiftf0-bundled");
    write_tone(&scratch, "bank/A2.wav");

    let run = cli(&scratch.0, &["bank", "--estimator", "swiftf0", "-v"]);
    assert_eq!(run.code, 0, "{}", run.stderr);
    assert!(run.stderr.contains("wrote frq"), "{}", run.stderr);

    let table = frq::read(&scratch.path("bank/A2_wav.frq")).unwrap();
    assert_eq!(table.sample_rate, 44_100);
    assert_eq!(table.hop_samples, 256);
    for &value in &table.f0_hz {
        assert!(value.is_finite(), "non-finite f0: {value}");
        assert!((71.0..=800.0).contains(&value) || value == 0.0, "{value}");
    }
    assert_eq!(*table.f0_hz.last().unwrap(), 0.0, "the trailing rule");
}

/// The default resolves to RMVPE when the modern build finds a model, and to
/// DIO otherwise (#49); an explicit choice always wins.
#[test]
fn the_default_estimator_is_capability_aware() {
    if !cfg!(feature = "ml") {
        // The compat build has no ML tier, so the default is DIO by
        // construction and there is nothing model-dependent to exercise.
        return;
    }
    let scratch = Scratch::new("default-estimator");
    write_tone(&scratch, "bank/A2.wav");
    let model_dir = fixture_model_dir(&scratch);

    // With the model available the default run must use it. The fixture's
    // f0 values (~441.5 / ~220.4 Hz alternating) are not what DIO finds on
    // the 220 Hz tone, so the two tables must differ.
    let with_model = run_ok_with_env(
        &scratch.0,
        &["bank", "-v"],
        &[("KIRAFRQ_ML_DIR", &model_dir)],
    );
    assert!(
        with_model.stderr.contains("wrote frq"),
        "{}",
        with_model.stderr
    );
    let default_table = frq::read(&scratch.path("bank/A2_wav.frq")).unwrap();

    // An explicit `--estimator dio` still wins with the model present.
    let explicit = run_ok_with_env(
        &scratch.0,
        &["bank", "--overwrite", "--estimator", "dio"],
        &[("KIRAFRQ_ML_DIR", &model_dir)],
    );
    assert!(explicit.stderr.contains("written 1"), "{}", explicit.stderr);
    let dio_table = frq::read(&scratch.path("bank/A2_wav.frq")).unwrap();

    assert_ne!(
        default_table.f0_hz, dio_table.f0_hz,
        "the default must have used RMVPE, not DIO"
    );
    let fixture_voiced: Vec<f64> = default_table
        .f0_hz
        .iter()
        .copied()
        .filter(|&value| value > 0.0)
        .collect();
    assert!(
        fixture_voiced.iter().any(|&value| value > 400.0),
        "the fixture's ~441.5 Hz slot must appear: {fixture_voiced:?}"
    );
}

fn run_ok_with_env(dir: &Path, args: &[&str], env: &[(&str, &str)]) -> Run {
    let run = cli_with_env(dir, args, env);
    assert_eq!(run.code, 0, "args {args:?}: {}", run.stderr);
    assert!(
        run.stdout_is_empty(),
        "stdout must stay empty: {:?}",
        run.stdout
    );
    run
}

// --- codepage sharing flag --------------------------------------------------

#[test]
fn ensure_japanese_codepage_warns_when_mrq_is_not_selected() {
    let scratch = Scratch::new("codepage");
    scratch.write("bank/A2.wav", b"placeholder");

    let run = run_ok(
        &scratch.0,
        &["bank", "--dry-run", "--ensure-japanese-codepage"],
    );
    assert!(run.stderr.contains("has no effect"), "{}", run.stderr);
}

#[cfg(not(windows))]
#[test]
fn ensure_japanese_codepage_is_a_warned_no_op_off_windows() {
    let scratch = Scratch::new("codepage-unix");
    scratch.write("bank/A2.wav", b"placeholder");

    let run = run_ok(
        &scratch.0,
        &[
            "bank",
            "--dry-run",
            "--format",
            "mrq",
            "--ensure-japanese-codepage",
        ],
    );
    assert!(
        run.stderr.contains("no-op on this platform"),
        "{}",
        run.stderr
    );
}

// --- non-ascii paths --------------------------------------------------------

#[test]
fn non_ascii_wav_names_round_trip() {
    let scratch = Scratch::new("non-ascii");
    write_tone(&scratch, "bank/_かきくけこ.wav");

    let run = run_ok(&scratch.0, &["bank", "-v"]);
    assert!(
        run.stderr.contains("_かきくけこ.wav: wrote frq"),
        "{}",
        run.stderr
    );
    assert!(scratch.path("bank/_かきくけこ_wav.frq").exists());
}
