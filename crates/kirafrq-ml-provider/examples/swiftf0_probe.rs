//! Throwaway prototype probe for wayfinder #62: run SwiftF0 over real
//! voicebank wavs through the provider's real model-free seam
//! (`resample_to_model_rate`, `UvPolicy`, `map_to_table_grid`) and dump both
//! the native-grid pitch/confidence and the mapped table f0 for analysis.
//!
//! Not a product surface. Delete with the prototype branch.
//!
//! Run:
//! `cargo run -p kirafrq-ml-provider --features ort --example swiftf0_probe -- \
//!      <swiftf0.onnx> <out_dir> <wav>...`

use std::fmt::Write as _;
use std::io::Write as _;

use kirafrq_ml_provider::grid::{
    ModelContract, NativeContour, TABLE_HOP_SAMPLES, map_to_table_grid,
};
use kirafrq_ml_provider::policy::UvPolicy;
use kirafrq_ml_provider::resample::resample_to_model_rate;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;

/// SwiftF0's native time base (#59): 16 kHz, hop 256, +127.5-sample origin.
const SWIFTF0_CONTRACT: ModelContract = ModelContract {
    sample_rate: 16_000,
    hop_samples: 256,
    origin_s: 127.5 / 16_000.0,
};

/// The paper's "approximately 90%" default (#59), parameterized in #53.
const SWIFTF0_DEFAULT_CONFIDENCE: f64 = 0.9;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: swiftf0_probe <swiftf0.onnx> <out_dir> <wav>...");
        std::process::exit(2);
    }
    let model = &args[1];
    let out_dir = std::path::PathBuf::from(&args[2]);
    std::fs::create_dir_all(&out_dir).expect("create out dir");

    // Optional RMVPE comparison model: `<out_dir> [--rmvpe <rmvpe.onnx>] ...`.
    let mut inputs: Vec<String> = args[3..].to_vec();
    let rmvpe_model = if inputs.first().map(String::as_str) == Some("--rmvpe") {
        let model = inputs.get(1).cloned().expect("--rmvpe needs a path");
        inputs.drain(0..2);
        Some(model)
    } else {
        None
    };
    let rmvpe = rmvpe_model.map(|path| {
        kirafrq_ml_provider::rmvpe::Rmvpe::new(
            path.into(),
            UvPolicy {
                confidence_threshold: kirafrq_ml_provider::RMVPE_DEFAULT_CONFIDENCE_THRESHOLD,
                floor_hz: 71.0,
                ceiling_hz: 800.0,
            },
            0,
        )
    });

    // Expand directory arguments recursively; passing dirs avoids the shell
    // mangling the corpus's non-ASCII filenames.
    let mut wavs: Vec<std::path::PathBuf> = Vec::new();
    for arg in &inputs {
        let path = std::path::PathBuf::from(arg);
        if path.is_dir() {
            collect_wavs(&path, &mut wavs);
        } else {
            wavs.push(path);
        }
    }
    wavs.sort();
    eprintln!("{} wavs", wavs.len());

    let mut session = build_session(model);
    let (pitch_index, confidence_index) = output_indices(&session);
    eprintln!("outputs: pitch index {pitch_index}, confidence index {confidence_index}");

    let policy = UvPolicy {
        confidence_threshold: SWIFTF0_DEFAULT_CONFIDENCE,
        floor_hz: 71.0,
        ceiling_hz: 800.0,
    };

    let total = wavs.len();
    for (done, wav) in wavs.iter().enumerate() {
        match run_one(
            &mut session,
            pitch_index,
            confidence_index,
            &policy,
            rmvpe.as_ref(),
            &wav.to_string_lossy(),
            &out_dir,
        ) {
            Ok(()) => {}
            Err(error) => eprintln!("{}: FAILED: {error}", wav.display()),
        }
        if (done + 1) % 25 == 0 {
            eprintln!("  ... {}/{total}", done + 1);
        }
    }
}

fn collect_wavs(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_wavs(&path, out);
        } else if path
            .extension()
            .is_some_and(|extension| extension.eq_ignore_ascii_case("wav"))
        {
            out.push(path);
        }
    }
}

fn build_session(model: &str) -> Session {
    Session::builder()
        .expect("session builder")
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .expect("optimization level")
        .with_inter_threads(1)
        .expect("inter threads")
        .commit_from_file(model)
        .unwrap_or_else(|error| panic!("cannot load {model}: {error}"))
}

/// The model emits `pitch_hz` and `confidence`; index them by name rather than
/// trusting the export order.
fn output_indices(session: &Session) -> (usize, usize) {
    let names: Vec<String> = session
        .outputs()
        .iter()
        .map(|output| output.name().to_string())
        .collect();
    let find = |want: &str, fallback: usize| {
        names
            .iter()
            .position(|name| name == want)
            .unwrap_or(fallback)
    };
    (find("pitch_hz", 0), find("confidence", 1))
}

fn run_one(
    session: &mut Session,
    pitch_index: usize,
    confidence_index: usize,
    policy: &UvPolicy,
    rmvpe: Option<&kirafrq_ml_provider::rmvpe::Rmvpe>,
    wav: &str,
    out_dir: &std::path::Path,
) -> Result<(), String> {
    let path = std::path::Path::new(wav);
    let decoded = match kirafrq_audio::decode_wav(path) {
        kirafrq_audio::DecodeOutcome::Decoded(decoded) => decoded,
        kirafrq_audio::DecodeOutcome::Empty => return Err("empty wav".to_string()),
        kirafrq_audio::DecodeOutcome::Failed(reason) => return Err(reason),
    };
    let samples = decoded.samples;

    // The #47 short-clip short-circuit, exactly as the provider does it.
    if samples.len() < TABLE_HOP_SAMPLES {
        let track = map_to_table_grid(
            &NativeContour {
                contract: SWIFTF0_CONTRACT,
                f0_hz: Vec::new(),
            },
            samples.len(),
        );
        return write_dump(
            out_dir,
            path,
            samples.len(),
            0,
            &[],
            &[],
            &[],
            &track.f0_hz,
            None,
        );
    }

    let audio16 = resample_to_model_rate(&samples, 44_100, 16_000).map_err(|e| e.to_string())?;
    let audio16: Vec<f32> = audio16.into_iter().map(|s| s as f32).collect();
    let samples_16k = audio16.len();

    // RMVPE on the exact same 16 kHz input, through the provider's own path.
    let rmvpe_dump = match rmvpe {
        Some(estimator) => {
            let native = estimator
                .native_estimate(&audio16)
                .map_err(|e| e.to_string())?;
            let track = map_to_table_grid(
                &NativeContour {
                    contract: kirafrq_ml_provider::grid::RMVPE_CONTRACT,
                    f0_hz: native.clone(),
                },
                samples.len(),
            );
            Some((native, track.f0_hz))
        }
        None => None,
    };

    // #59: a mandatory zero-pad to >= 256 samples; shorter inputs fail inside
    // ONNX Runtime's STFT kernel.
    let mut padded = audio16;
    if padded.len() < 256 {
        padded.resize(256, 0.0);
    }

    let input = Tensor::from_array(([1_i64, padded.len() as i64], padded.into_boxed_slice()))
        .map_err(|e| e.to_string())?;
    let outputs = session
        .run(ort::inputs![input])
        .map_err(|e| format!("inference failed: {e}"))?;

    let (_, pitch) = outputs[pitch_index]
        .try_extract_tensor::<f32>()
        .map_err(|e| e.to_string())?;
    let (_, confidence) = outputs[confidence_index]
        .try_extract_tensor::<f32>()
        .map_err(|e| e.to_string())?;
    if pitch.len() != confidence.len() {
        return Err(format!(
            "pitch/confidence length mismatch: {} vs {}",
            pitch.len(),
            confidence.len()
        ));
    }
    let native_pitch: Vec<f64> = pitch.iter().map(|&v| f64::from(v)).collect();
    let native_conf: Vec<f64> = confidence.iter().map(|&v| f64::from(v)).collect();
    let native_f0: Vec<f64> = native_pitch
        .iter()
        .zip(&native_conf)
        .map(|(&f0, &conf)| policy.apply(f0, conf))
        .collect();

    let track = map_to_table_grid(
        &NativeContour {
            contract: SWIFTF0_CONTRACT,
            f0_hz: native_f0.clone(),
        },
        samples.len(),
    );

    write_dump(
        out_dir,
        path,
        samples.len(),
        samples_16k,
        &native_pitch,
        &native_conf,
        &native_f0,
        &track.f0_hz,
        rmvpe_dump
            .as_ref()
            .map(|(native, table)| (native.as_slice(), table.as_slice())),
    )
}

#[allow(clippy::too_many_arguments)]
fn write_dump(
    out_dir: &std::path::Path,
    wav: &std::path::Path,
    samples: usize,
    samples_16k: usize,
    native_pitch: &[f64],
    native_conf: &[f64],
    native_f0: &[f64],
    table_f0: &[f64],
    rmvpe: Option<(&[f64], &[f64])>,
) -> Result<(), String> {
    let stem = wav
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or("no file stem")?;
    let out = out_dir.join(format!("{stem}.swift.json"));
    let mut body = String::with_capacity(table_f0.len() * 16);
    let _ = writeln!(body, "{{");
    let _ = writeln!(body, "\"samples\": {samples},");
    let _ = writeln!(body, "\"samples_16k\": {samples_16k},");
    let _ = writeln!(body, "\"origin_s\": {:.10},", SWIFTF0_CONTRACT.origin_s);
    let _ = writeln!(body, "\"hop\": {},", SWIFTF0_CONTRACT.hop_samples);
    let _ = writeln!(body, "\"table_period_ms\": {:.10},", 256.0 / 44100.0 * 1000.0);
    array(&mut body, "native_pitch", native_pitch);
    array(&mut body, "native_conf", native_conf);
    array(&mut body, "native_f0", native_f0);
    array(&mut body, "table_f0", table_f0);
    match rmvpe {
        Some((native, table)) => {
            array(&mut body, "rmvpe_native_f0", native);
            array(&mut body, "rmvpe_table_f0", table);
        }
        None => {
            array(&mut body, "rmvpe_native_f0", &[]);
            array(&mut body, "rmvpe_table_f0", &[]);
        }
    }
    // trailing comma cleanup: replace the last comma before the closing brace
    if body.trim_end().ends_with(',') {
        let trimmed = body.trim_end().trim_end_matches(',').to_string();
        body = trimmed;
        body.push('\n');
    }
    body.push('}');
    std::fs::File::create(&out)
        .and_then(|mut file| file.write_all(body.as_bytes()))
        .map_err(|e| format!("cannot write {}: {e}", out.display()))
}

fn array(body: &mut String, name: &str, values: &[f64]) {
    let _ = write!(body, "\"{name}\": [");
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            body.push(',');
        }
        if value.is_finite() {
            let _ = write!(body, "{value:.6}");
        } else {
            body.push_str("null");
        }
    }
    body.push_str("],\n");
}
