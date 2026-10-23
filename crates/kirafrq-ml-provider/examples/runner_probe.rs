//! Local verification probe (not a CI test): loads an ML model under each ML
//! runner this build has (CPU, DirectML, Core ML) and reports, per runner,
//! the session build time, the first call at each new input length, the warm
//! time per audio-second, and the f0 drift against the CPU reference.
//!
//! SwiftF0 always loads from the bundled model; pass `--rmvpe <rmvpe.onnx>`
//! to also probe RMVPE. Audio is a set of synthetic harmonic stacks of
//! different lengths unless `--wav` points at a real 44.1 kHz wav (then that
//! wav and its leading 60% and 30% are the lengths).
//!
//! Run:
//! `cargo run --release -p kirafrq-ml-provider --features ort --example runner_probe -- \
//!      [--rmvpe <rmvpe.onnx>] [--wav <file.wav>]`

use std::path::PathBuf;
use std::time::{Duration, Instant};

use kirafrq_ml_provider::model::ModelSource;
use kirafrq_ml_provider::policy::UvPolicy;
use kirafrq_ml_provider::resample::resample_to_model_rate;
use kirafrq_ml_provider::runner::{Runner, SessionConfig};
use kirafrq_ml_provider::swiftf0::{BUNDLED_MODEL, SwiftF0};
use kirafrq_ml_provider::{COREML_SUPPORTED, DIRECTML_SUPPORTED, Error};

/// The model rate every input is resampled to.
const MODEL_RATE: f64 = 16_000.0;

/// Synthetic clip lengths in seconds, spanning a typical voicebank's wavs.
const SYNTHETIC_SECONDS: [f64; 4] = [0.9, 2.1, 3.4, 5.2];

/// Warm repeats per length.
const WARM_REPEATS: u32 = 3;

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
                eprintln!("usage: runner_probe [--rmvpe <rmvpe.onnx>] [--wav <file.wav>]");
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
    eprintln!("coreml supported in this build: {COREML_SUPPORTED}");

    let clips = probe_audio(wav_path.as_ref());
    let mut runners = vec![("cpu", Runner::Cpu)];
    if DIRECTML_SUPPORTED {
        runners.push(("directml", Runner::DirectMl));
    }
    if COREML_SUPPORTED {
        runners.push(("coreml", Runner::CoreMl));
    }

    if let Some(path) = rmvpe_path {
        let build = |runner| {
            kirafrq_ml_provider::rmvpe::Rmvpe::new(path.clone(), UvPolicy::rmvpe(None), 0)
                .with_session_config(SessionConfig {
                    runner,
                    directml_device: None,
                })
        };
        probe("RMVPE", &runners, &clips, build);
    }
    let build = |runner| {
        SwiftF0::new(
            ModelSource::Bundled {
                file_name: "swiftf0.onnx",
                bytes: BUNDLED_MODEL,
            },
            UvPolicy::swiftf0(None),
            0,
        )
        .with_session_config(SessionConfig {
            runner,
            directml_device: None,
        })
    };
    probe("SwiftF0", &runners, &clips, build);
}

/// The 16 kHz f32 clips: a decoded wav and two prefixes of it, or harmonic
/// stacks of [`SYNTHETIC_SECONDS`] with a gliding pitch.
fn probe_audio(wav_path: Option<&PathBuf>) -> Vec<Vec<f32>> {
    let to_16k = |samples_44k: &[f64]| -> Vec<f32> {
        resample_to_model_rate(samples_44k, 44_100, 16_000)
            .expect("resample")
            .into_iter()
            .map(|sample| sample as f32)
            .collect()
    };
    match wav_path {
        Some(path) => {
            let samples = match kirafrq_audio::decode_wav(path) {
                kirafrq_audio::DecodeOutcome::Decoded(decoded) => decoded.samples,
                other => {
                    eprintln!("{}: cannot decode ({other:?})", path.display());
                    std::process::exit(1);
                }
            };
            [1.0, 0.6, 0.3]
                .iter()
                .map(|share| to_16k(&samples[..(samples.len() as f64 * share) as usize]))
                .collect()
        }
        None => SYNTHETIC_SECONDS
            .iter()
            .map(|seconds| {
                const RATE: f64 = 44_100.0;
                let len = (RATE * seconds) as usize;
                let mut phase = 0.0_f64;
                let samples: Vec<f64> = (0..len)
                    .map(|n| {
                        // A one-octave glide from 165 Hz, so every RMVPE bin
                        // region sees signal.
                        let hz = 165.0 * 2f64.powf(n as f64 / len as f64);
                        phase += std::f64::consts::TAU * hz / RATE;
                        (1..=10)
                            .map(|harmonic| (phase * harmonic as f64).sin() / harmonic as f64)
                            .sum::<f64>()
                            * 0.3
                    })
                    .collect();
                to_16k(&samples)
            })
            .collect(),
    }
}

fn probe<E>(name: &str, runners: &[(&str, Runner)], clips: &[Vec<f32>], build: impl Fn(Runner) -> E)
where
    E: NativeEstimator,
{
    let audio_seconds: f64 = clips
        .iter()
        .map(|clip| clip.len() as f64 / MODEL_RATE)
        .sum();
    let mut reference: Option<Vec<Vec<f64>>> = None;
    for &(label, runner) in runners {
        let estimator = build(runner);
        let start = Instant::now();
        if let Err(error) = estimator.ensure_session() {
            println!("{name} {label}: session failed: {error}");
            continue;
        }
        let session = start.elapsed();

        let mut firsts = Vec::new();
        let mut outputs = Vec::new();
        for clip in clips {
            let start = Instant::now();
            match estimator.native_estimate(clip) {
                Ok(f0) => outputs.push(f0),
                Err(error) => {
                    println!("{name} {label}: inference failed: {error}");
                    return;
                }
            }
            firsts.push(start.elapsed());
        }
        let mut warm = Duration::ZERO;
        for clip in clips {
            let start = Instant::now();
            for _ in 0..WARM_REPEATS {
                if let Err(error) = estimator.native_estimate(clip) {
                    println!("{name} {label}: repeat inference failed: {error}");
                    return;
                }
            }
            warm += start.elapsed() / WARM_REPEATS;
        }

        // First call at a new length minus that length's warm cost is the
        // per-length stall.
        let firsts: Vec<String> = firsts
            .iter()
            .zip(clips)
            .map(|(first, clip)| {
                format!(
                    "{:.1}s {:.0} ms",
                    clip.len() as f64 / MODEL_RATE,
                    ms(*first)
                )
            })
            .collect();
        println!(
            "{name} {label}: session {:.0} ms; first call {}; warm {:.1} ms per audio-second",
            ms(session),
            firsts.join(", "),
            ms(warm) / audio_seconds
        );
        match &reference {
            None => reference = Some(outputs),
            Some(reference) => println!("{name} {label}: {}", drift(reference, &outputs)),
        }
    }
}

/// The largest f0 difference against the CPU reference over frames both call
/// voiced, in Hz and cents, plus the voiced/unvoiced disagreements.
fn drift(reference: &[Vec<f64>], outputs: &[Vec<f64>]) -> String {
    let mut max_hz = 0.0_f64;
    let mut max_cents = 0.0_f64;
    let mut flips = 0;
    let mut frames = 0;
    for (reference, output) in reference.iter().zip(outputs) {
        for (&a, &b) in reference.iter().zip(output) {
            frames += 1;
            if (a > 0.0) != (b > 0.0) {
                flips += 1;
            } else if a > 0.0 {
                max_hz = max_hz.max((a - b).abs());
                max_cents = max_cents.max(1_200.0 * (b / a).log2().abs());
            }
        }
    }
    format!(
        "vs cpu max |Δf0| {max_hz:.6} Hz ({max_cents:.4} cents), {flips} of {frames} frames flip voicing"
    )
}

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
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
