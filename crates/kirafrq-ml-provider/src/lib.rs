//! ML f0 estimation for KiraFrqGen: the model-free resample/grid layer and
//! the RMVPE model path over ONNX Runtime.
//!
//! The crate splits in two by the `ort` feature, so the model-free half
//! compiles and tests in every configuration (#48/#56):
//!
//! - **Model-free** (always compiled): the 44.1 kHz -> 16 kHz resample
//!   ([`resample_to_model_rate`]; #47), the native-contour -> 5.805 ms
//!   table-grid mapping ([`map_to_table_grid`]; #47), the native-grid
//!   voiced/unvoiced policy ([`UvPolicy`]; #53) and model-file resolution
//!   ([`resolve_model`]; #48).
//! - **RMVPE** (`ort` on): the ONNX Runtime session, log-mel frontend and
//!   salience decoder, under [`rmvpe`]. Every `ort` call lives there.
//!
//! The pipeline talks to this crate only through `kirafrqgen-core`'s
//! `F0Estimator` seam; this crate never depends on core.

pub mod grid;
pub mod model;
pub mod policy;
pub mod resample;

#[cfg(feature = "ort")]
#[doc(hidden)]
pub mod mel;
#[cfg(feature = "ort")]
pub mod rmvpe;

pub use grid::{
    ModelContract, NativeContour, RMVPE_CONTRACT, Track, map_to_table_grid, table_frame_count,
};
pub use model::{MODEL_DIR_ENV, MODEL_FILE_NAME, ModelError, resolve_model};
pub use policy::{RMVPE_DEFAULT_CONFIDENCE_THRESHOLD, UvPolicy};
pub use resample::resample_to_model_rate;

/// Coarse per-file progress out of an ML estimate (#34/#48): one tick per
/// stage, `done` of `total`. `kirafrqgen-core` adapts its frame observer onto
/// this so the ML path adds no progress machinery of its own.
pub trait ProgressObserver: Send + Sync {
    fn report(&self, done: usize, total: usize);
}

/// Errors from the model-free layer and the RMVPE path.
#[derive(Debug)]
pub enum Error {
    /// Resampling the pipeline audio to the model rate failed.
    Resample(String),
    /// The model file could not be resolved; see [`ModelError`].
    Model(ModelError),
    /// ONNX Runtime failed: session create, inference or a model whose I/O
    /// contract is not the RMVPE one.
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
