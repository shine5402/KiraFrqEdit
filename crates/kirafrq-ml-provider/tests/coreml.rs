//! The Core ML runner (#84), Apple-silicon macOS only: SwiftF0 under Core ML
//! matches the CPU runner, and the Core ML model cache is keyed by the model's
//! content, so a rebuilt session reuses it and a replaced model compiles
//! fresh. RMVPE is checked locally with the `runner_probe` example.

#![cfg(all(feature = "ort", target_os = "macos", target_arch = "aarch64"))]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use kirafrq_ml_provider::coreml_cache_dir;
use kirafrq_ml_provider::model::ModelSource;
use kirafrq_ml_provider::policy::UvPolicy;
use kirafrq_ml_provider::runner::{Runner, SessionConfig};
use kirafrq_ml_provider::swiftf0::{BUNDLED_MODEL, SwiftF0};

const COREML: SessionConfig = SessionConfig {
    runner: Runner::CoreMl,
    directml_device: None,
};

fn swiftf0(source: ModelSource, jobs: usize, session_config: SessionConfig) -> SwiftF0 {
    SwiftF0::new(source, UvPolicy::swiftf0(None), jobs).with_session_config(session_config)
}

fn bundled() -> ModelSource {
    ModelSource::Bundled {
        file_name: "swiftf0.onnx",
        bytes: BUNDLED_MODEL,
    }
}

/// Voiced stretches at several pitches between silences, at 16 kHz.
fn phrase_16k() -> Vec<f32> {
    let mut audio = vec![0.0_f32; 4_000];
    for hz in [110.0, 196.0, 330.0, 523.0] {
        audio.extend((0..12_000).map(|index| {
            let t = index as f64 / 16_000.0;
            let value: f64 = (1..=10)
                .map(|harmonic| {
                    (std::f64::consts::TAU * hz * harmonic as f64 * t).sin() / harmonic as f64
                })
                .sum();
            (value * 0.1) as f32
        }));
        audio.extend(std::iter::repeat_n(0.0, 4_000));
    }
    audio
}

#[test]
fn swiftf0_under_coreml_matches_the_cpu() {
    let audio = phrase_16k();
    let cpu = swiftf0(bundled(), 1, SessionConfig::default())
        .native_estimate(&audio)
        .unwrap();
    let coreml = swiftf0(bundled(), 1, COREML)
        .native_estimate(&audio)
        .unwrap();

    assert_eq!(cpu.len(), coreml.len());
    let mut voiced = 0;
    for (frame, (&reference, &value)) in cpu.iter().zip(&coreml).enumerate() {
        assert_eq!(
            reference > 0.0,
            value > 0.0,
            "frame {frame}: voicing flipped ({reference} Hz vs {value} Hz)"
        );
        if reference > 0.0 {
            voiced += 1;
            let cents = 1_200.0 * (value / reference).log2().abs();
            assert!(
                cents < 1.0,
                "frame {frame}: {reference} Hz vs {value} Hz is {cents} cents"
            );
        }
    }
    assert!(
        voiced > cpu.len() / 2,
        "the phrase is mostly voiced: {voiced}"
    );
}

#[test]
fn the_model_cache_is_reused_and_a_replaced_model_builds_fresh() {
    let scratch = Scratch::new();
    let path = scratch.0.join("swiftf0.onnx");
    let source = ModelSource::File(path.clone());

    std::fs::write(&path, unique_model("first")).unwrap();
    let first_dir = coreml_cache_dir(&source).unwrap();
    swiftf0(source.clone(), 1, COREML).ensure_session().unwrap();
    let first = snapshot(&first_dir);
    assert!(
        !first.is_empty(),
        "the first build fills {}",
        first_dir.display()
    );

    // A different thread count rebuilds the session from the same model.
    swiftf0(source.clone(), 2, COREML).ensure_session().unwrap();
    assert_eq!(coreml_cache_dir(&source).unwrap(), first_dir);
    assert_eq!(snapshot(&first_dir), first, "the rebuild reuses the cache");

    // The same path with different bytes is a different model.
    std::fs::write(&path, unique_model("second")).unwrap();
    let second_dir = coreml_cache_dir(&source).unwrap();
    assert_ne!(second_dir, first_dir);
    swiftf0(source.clone(), 1, COREML).ensure_session().unwrap();
    assert!(
        !snapshot(&second_dir).is_empty(),
        "the replaced model compiles"
    );
    assert_eq!(snapshot(&first_dir), first, "the old compile is left alone");

    for dir in [first_dir, second_dir] {
        let _ = std::fs::remove_dir_all(dir);
    }
}

/// The bundled SwiftF0 with a run-unique `doc_string` appended. Protobuf
/// keeps the last value of a repeated scalar field, so this stays the same
/// graph under new bytes, and no earlier test run's cache can match it.
fn unique_model(tag: &str) -> Vec<u8> {
    let nanos = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let doc = format!(
        "kirafrq coreml cache test {tag} {} {nanos}",
        std::process::id()
    );
    assert!(doc.len() < 128, "one-byte varint length");
    let mut bytes = BUNDLED_MODEL.to_vec();
    // ModelProto.doc_string is field 6, length-delimited.
    bytes.push((6 << 3) | 2);
    bytes.push(doc.len() as u8);
    bytes.extend_from_slice(doc.as_bytes());
    bytes
}

/// Every file under `dir` with its modification time; empty when `dir` does
/// not exist.
fn snapshot(dir: &Path) -> BTreeMap<PathBuf, SystemTime> {
    let mut files = BTreeMap::new();
    let mut pending = vec![dir.to_path_buf()];
    while let Some(dir) = pending.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let metadata = entry.metadata().unwrap();
            if metadata.is_dir() {
                pending.push(path);
            } else {
                files.insert(path, metadata.modified().unwrap());
            }
        }
    }
    files
}

struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Self {
        let dir = std::env::temp_dir().join(format!(
            "kirafrq-ml-provider-coreml-cache-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
