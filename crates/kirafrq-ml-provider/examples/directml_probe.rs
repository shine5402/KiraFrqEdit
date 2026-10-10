//! Local verification probe (not a CI test): loads an ML model under the CPU
//! and the DirectML runners and reports, per runner, whether the session
//! builds, how the native f0 compares to the CPU reference and how long one
//! inference takes.
//!
//! SwiftF0 always loads from the bundled model; pass `--rmvpe <rmvpe.onnx>`
//! to also probe RMVPE. Audio is a synthetic harmonic stack unless `--wav`
//! points at a real 44.1 kHz wav.
//!
//! Run:
//! `cargo run -p kirafrq-ml-provider --features ort --example directml_probe -- \
//!      [--rmvpe <rmvpe.onnx>] [--wav <file.wav>]`

use std::path::PathBuf;
use std::time::Instant;

use kirafrq_ml_provider::model::ModelSource;
use kirafrq_ml_provider::policy::UvPolicy;
use kirafrq_ml_provider::resample::resample_to_model_rate;
use kirafrq_ml_provider::runner::{Runner, SessionConfig};
use kirafrq_ml_provider::swiftf0::{BUNDLED_MODEL, SwiftF0};
use kirafrq_ml_provider::{DIRECTML_SUPPORTED, Error};

const RUNNERS: [(&str, Runner); 2] = [("cpu", Runner::Cpu), ("directml", Runner::DirectMl)];

fn main() {
    let mut rmvpe_path = None;
    let mut wav_path = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--rmvpe" => rmvpe_path = args.next().map(PathBuf::from),
            "--wav" => wav_path = args.next().map(PathBuf::from),
            other => {
                eprintln!("unknown argument `{other}`");
                eprintln!("usage: directml_probe [--rmvpe <rmvpe.onnx>] [--wav <file.wav>]");
                std::process::exit(2);
            }
        }
    }
    eprintln!(
        "directml supported in this build: {DIRECTML_SUPPORTED} ({} adapter(s) listed)",
        kirafrq_ml_provider::directml_devices()
            .map(|devices| devices.len())
            .unwrap_or(0)
    );

    let audio_16k = probe_audio(wav_path.as_ref());
    if let Some(path) = rmvpe_path {
        let build = |runner| {
            kirafrq_ml_provider::rmvpe::Rmvpe::new(path.clone(), UvPolicy::rmvpe(None), 1)
                .with_session_config(SessionConfig {
                    runner,
                    directml_device: None,
                })
        };
        probe("RMVPE", &audio_16k, build);
    }
    let build = |runner| {
        SwiftF0::new(
            ModelSource::Bundled {
                file_name: "swiftf0.onnx",
                bytes: BUNDLED_MODEL,
            },
            UvPolicy::swiftf0(None),
            1,
        )
        .with_session_config(SessionConfig {
            runner,
            directml_device: None,
        })
    };
    probe("SwiftF0", &audio_16k, build);
}

/// The 16 kHz f32 input: a decoded wav or a 3 s harmonic stack.
fn probe_audio(wav_path: Option<&PathBuf>) -> Vec<f32> {
    let samples_44k: Vec<f64> = match wav_path {
        Some(path) => match kirafrq_audio::decode_wav(path) {
            kirafrq_audio::DecodeOutcome::Decoded(decoded) => decoded.samples,
            other => {
                eprintln!("{}: cannot decode ({other:?})", path.display());
                std::process::exit(1);
            }
        },
        None => {
            const RATE: f64 = 44_100.0;
            (0..(RATE * 3.0) as usize)
                .map(|n| {
                    let t = n as f64 / RATE;
                    (1..=10)
                        .map(|harmonic| {
                            (2.0 * std::f64::consts::PI * 220.0 * harmonic as f64 * t).sin()
                                / harmonic as f64
                        })
                        .sum::<f64>()
                        * 0.3
                })
                .collect()
        }
    };
    resample_to_model_rate(&samples_44k, 44_100, 16_000)
        .expect("resample")
        .into_iter()
        .map(|sample| sample as f32)
        .collect()
}

fn probe<E>(name: &str, audio_16k: &[f32], build: impl Fn(Runner) -> E)
where
    E: NativeEstimator,
{
    let mut reference: Option<Vec<f64>> = None;
    for (label, runner) in RUNNERS {
        let estimator = build(runner);
        if let Err(error) = estimator.ensure_session() {
            println!("{name} {label}: session failed: {error}");
            continue;
        }
        let start = Instant::now();
        let f0 = match estimator.native_estimate(audio_16k) {
            Ok(f0) => f0,
            Err(error) => {
                println!("{name} {label}: inference failed: {error}");
                continue;
            }
        };
        let first = start.elapsed();
        let start = Instant::now();
        for _ in 0..3 {
            if let Err(error) = estimator.native_estimate(audio_16k) {
                println!("{name} {label}: repeat inference failed: {error}");
                break;
            }
        }
        let repeats = start.elapsed() / 3;

        let delta = reference.as_ref().map(|reference| {
            reference
                .iter()
                .zip(&f0)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max)
        });
        match (&reference, delta) {
            (None, _) => {
                println!(
                    "{name} {label}: {} frames, first {first:?}, min/rep {repeats:?} (reference)",
                    f0.len()
                );
                reference = Some(f0);
            }
            (Some(_), Some(delta)) => println!(
                "{name} {label}: {} frames, first {first:?}, min/rep {repeats:?}, max |Δf0| {delta:.6} Hz",
                f0.len()
            ),
            (Some(_), None) => unreachable!("delta is computed for every non-reference run"),
        }
    }
}

/// The subset of the estimator API the probe needs; implemented for both
/// model paths so the loop stays generic.
pub trait NativeEstimator {
    fn ensure_session(&self) -> Result<(), Error>;
    fn native_estimate(&self, audio_16k: &[f32]) -> Result<Vec<f64>, Error>;
}

impl NativeEstimator for kirafrq_ml_provider::rmvpe::Rmvpe {
    fn ensure_session(&self) -> Result<(), Error> {
        kirafrq_ml_provider::rmvpe::Rmvpe::ensure_session(self)
    }

    fn native_estimate(&self, audio_16k: &[f32]) -> Result<Vec<f64>, Error> {
        kirafrq_ml_provider::rmvpe::Rmvpe::native_estimate(self, audio_16k)
    }
}

impl NativeEstimator for SwiftF0 {
    fn ensure_session(&self) -> Result<(), Error> {
        SwiftF0::ensure_session(self)
    }

    fn native_estimate(&self, audio_16k: &[f32]) -> Result<Vec<f64>, Error> {
        SwiftF0::native_estimate(self, audio_16k)
    }
}
