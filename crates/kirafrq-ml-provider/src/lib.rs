//! ML f0 estimation for KiraFrqGen: the model-free resample/grid layer and
//! the RMVPE and SwiftF0 model paths over ONNX Runtime.
//!
//! The crate splits in two by the `ort` feature, so the model-free half
//! compiles and tests in every configuration (#48/#56):
//!
//! - **Model-free** (always compiled): the 44.1 kHz -> 16 kHz resample
//!   ([`resample_to_model_rate`]; #47), the native-contour -> 5.805 ms
//!   table-grid mapping ([`map_to_table_grid`]; #47), the native-grid
//!   voiced/unvoiced policy ([`UvPolicy`]; #53), model-file resolution
//!   ([`resolve_model`]/[`resolve_file`]; #48/#69) and the ML runner
//!   selection with the DirectML adapter list and the Core ML check
//!   ([`runner`]).
//! - **Models** (`ort` on): the ONNX Runtime session, RMVPE's log-mel
//!   frontend and salience decoder under [`rmvpe`], and SwiftF0's raw-audio
//!   adapter under [`swiftf0`] (#69). Every `ort` call lives there.
//!
//! The pipeline talks to this crate only through `kirafrqgen-core`'s
//! `F0Estimator` seam; this crate never depends on core.

pub mod grid;
pub mod model;
pub mod policy;
pub mod resample;
pub mod runner;

#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "ort"))]
mod coreml;
#[cfg(feature = "ort")]
#[doc(hidden)]
pub mod mel;
#[cfg(feature = "ort")]
pub mod rmvpe;
#[cfg(feature = "ort")]
mod session;
#[cfg(feature = "ort")]
pub mod swiftf0;

#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "ort"))]
pub use coreml::cache_dir as coreml_cache_dir;
pub use grid::{
    ModelContract, NativeContour, RMVPE_CONTRACT, SWIFTF0_CONTRACT, Track, map_to_table_grid,
    table_frame_count,
};
pub use model::{
    MODEL_DIR_ENV, ModelError, ModelSource, RMVPE_FILE_NAME, SWIFTF0_FILE_NAME, resolve_file,
    resolve_model,
};
pub use policy::{
    RMVPE_DEFAULT_CONFIDENCE_THRESHOLD, SWIFTF0_DEFAULT_CONFIDENCE_THRESHOLD, UvPolicy,
};
pub use resample::resample_to_model_rate;
pub use runner::{
    COREML_SUPPORTED, DIRECTML_SUPPORTED, DeviceError, DirectMlDevice, Runner, SessionConfig,
    coreml_usable, directml_devices,
};

/// Coarse per-file progress out of an ML estimate (#34/#48): one tick per
/// stage, `done` of `total`. `kirafrqgen-core` adapts its frame observer onto
/// this so the ML path adds no progress machinery of its own.
pub trait ProgressObserver: Send + Sync {
    fn report(&self, done: usize, total: usize);
}

/// Errors from the model-free layer and the model paths.
#[derive(Debug)]
pub enum Error {
    /// Resampling the pipeline audio to the model rate failed.
    Resample(String),
    /// The model file could not be resolved; see [`ModelError`].
    Model(ModelError),
    /// ONNX Runtime failed: session create, inference or a model whose I/O
    /// contract is not the expected one.
    #[cfg(feature = "ort")]
    Ort(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Resample(message) => write!(f, "cannot resample to the model rate: {message}"),
            Error::Model(error) => error.fmt(f),
            #[cfg(feature = "ort")]
            Error::Ort(message) => write!(f, "ONNX Runtime: {message}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Model(error) => Some(error),
            _ => None,
        }
    }
}

impl From<ModelError> for Error {
    fn from(error: ModelError) -> Self {
        Error::Model(error)
    }
}
