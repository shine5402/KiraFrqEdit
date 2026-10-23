//! The Core ML runner's execution provider and its model cache (#84);
//! compiled only for Apple-silicon macOS `ort` builds.
//!
//! The options were settled by measurement on an M4 Max (see
//! `docs/research/coreml-runner.md`): the NeuralNetwork model format, whose
//! cost per new input length is ~5–10 ms where MLProgram re-specializes for
//! ~250–400 ms; CPU-and-GPU compute units, which keep the Neural Engine and
//! its accuracy loss on quiet inputs (ONNX Runtime #33319) out; dynamic
//! shapes; and no low-precision GPU accumulation.
//!
//! ONNX Runtime keys its Core ML compile cache by the model path and never
//! invalidates it, so each model gets its own cache subdirectory keyed by the
//! model's SHA-256, the ONNX Runtime build and the compute units.

use std::io::Read;
use std::path::PathBuf;
use std::sync::OnceLock;

use ort::ep::ExecutionProviderDispatch;
use ort::ep::coreml::{ComputeUnits, CoreML, ModelFormat};
use sha2::{Digest, Sha256};

use crate::Error;
use crate::model::{ModelSource, SWIFTF0_FILE_NAME};
use crate::runner::{Runner, SessionConfig};
use crate::session;
use crate::swiftf0::BUNDLED_MODEL;

const COMPUTE_UNITS: ComputeUnits = ComputeUnits::CPUAndGPU;

/// The Core ML execution provider for `source`. Registration failure is an
/// error, not the silent CPU fallback ONNX Runtime defaults to, so a session
/// that commits really runs Core ML.
pub(crate) fn execution_provider(source: &ModelSource) -> Result<ExecutionProviderDispatch, Error> {
    let cache = cache_dir(source)?;
    std::fs::create_dir_all(&cache).map_err(|error| cache_error(&cache, error))?;
    let cache = cache.to_str().ok_or_else(|| {
        Error::Ort(format!(
            "the Core ML model cache path {} is not UTF-8",
            cache.display()
        ))
    })?;
    Ok(CoreML::default()
        .with_model_format(ModelFormat::NeuralNetwork)
        .with_compute_units(COMPUTE_UNITS)
        .with_static_input_shapes(false)
        .with_low_precision_accumulation_on_gpu(false)
        .with_model_cache_dir(cache)
        .build()
        .error_on_failure())
}

/// The Core ML compile cache for `source`:
/// `~/Library/Caches/KiraFrqGen/coreml/<model SHA-256>/<ONNX Runtime build>-<compute units>`.
/// An on-disk model is hashed from its current bytes, so a model replaced at
/// the same path gets a fresh directory.
///
/// # Errors
///
/// [`Error::Ort`] when the model file cannot be read or there is no home
/// directory to cache under.
pub fn cache_dir(source: &ModelSource) -> Result<PathBuf, Error> {
    let model = match source {
        ModelSource::File(path) => {
            let read_error = |error| Error::Ort(format!("cannot read {}: {error}", path.display()));
            let mut file = std::fs::File::open(path).map_err(read_error)?;
            let mut hasher = Sha256::new();
            let mut buffer = vec![0; 1 << 20];
            loop {
                match file.read(&mut buffer).map_err(read_error)? {
                    0 => break,
                    read => hasher.update(&buffer[..read]),
                }
            }
            hasher.finalize()
        }
        ModelSource::Bundled { bytes, .. } => Sha256::digest(bytes),
    };
    let home = std::env::var_os("HOME")
        .filter(|home| !home.is_empty())
        .ok_or_else(|| Error::Ort("no HOME directory for the Core ML model cache".to_string()))?;
    // `ort::info()` names the ONNX Runtime release and commit.
    let runtime = Sha256::digest(ort::info().as_bytes());
    Ok(PathBuf::from(home)
        .join("Library/Caches/KiraFrqGen/coreml")
        .join(hex(&model))
        .join(format!("ort-{}-{COMPUTE_UNITS:?}", &hex(&runtime)[..16])))
}

/// Whether a Core ML session over the bundled SwiftF0 commits; probed once.
pub(crate) fn usable() -> bool {
    static USABLE: OnceLock<bool> = OnceLock::new();
    *USABLE.get_or_init(|| {
        let source = ModelSource::Bundled {
            file_name: SWIFTF0_FILE_NAME,
            bytes: BUNDLED_MODEL,
        };
        let config = SessionConfig {
            runner: Runner::CoreMl,
            directml_device: None,
        };
        session::build_session(&source, 1, config).is_ok()
    })
}

fn cache_error(dir: &std::path::Path, error: std::io::Error) -> Error {
    Error::Ort(format!(
        "cannot create the Core ML model cache {}: {error}",
        dir.display()
    ))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}
