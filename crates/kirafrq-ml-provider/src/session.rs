//! The process-wide ONNX Runtime session shared by every model path
//! (#48/#69): one session, built lazily behind a mutex and rebuilt only when
//! the model source, the thread count or the runner/device settings change.
//! An estimator can instead own its session ([`Owned`]; the ML server keeps
//! RMVPE and SwiftF0 loaded side by side). Either way, estimates run on a
//! session one at a time. Only [`build_session`] builds; the Core ML probe
//! uses it directly.

use std::sync::{Mutex, MutexGuard};

use ort::session::Session;
use ort::session::builder::{GraphOptimizationLevel, SessionBuilder};

use crate::Error;
use crate::model::ModelSource;
use crate::runner::{Runner, SessionConfig};

struct Cached {
    source: ModelSource,
    jobs: usize,
    config: SessionConfig,
    session: Session,
}

static SESSION: Mutex<Option<Cached>> = Mutex::new(None);

/// Run `f` against the session for `(source, jobs, config)`, building or
/// rebuilding it first when needed.
pub(crate) fn with<R>(
    source: &ModelSource,
    jobs: usize,
    config: SessionConfig,
    f: impl FnOnce(&mut Session) -> Result<R, Error>,
) -> Result<R, Error> {
    let mut guard: MutexGuard<'_, Option<Cached>> = SESSION
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let stale = guard.as_ref().is_none_or(|cached| {
        &cached.source != source || cached.jobs != jobs || cached.config != config
    });
    if stale {
        *guard = Some(build(source, jobs, config)?);
    }
    let cached = guard.as_mut().expect("just built");
    f(&mut cached.session)
}

fn build(source: &ModelSource, jobs: usize, config: SessionConfig) -> Result<Cached, Error> {
    Ok(Cached {
        source: source.clone(),
        jobs,
        config,
        session: build_session(source, jobs, config)?,
    })
}

/// A session one estimator owns, built up front.
pub(crate) struct Owned(Mutex<Session>);

impl Owned {
    pub(crate) fn build(
        source: &ModelSource,
        jobs: usize,
        config: SessionConfig,
    ) -> Result<Self, Error> {
        Ok(Self(Mutex::new(build_session(source, jobs, config)?)))
    }

    /// Run `f` on the session, waiting while another estimate uses it.
    pub(crate) fn run<R>(
        &self,
        f: impl FnOnce(&mut Session) -> Result<R, Error>,
    ) -> Result<R, Error> {
        let mut session = self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        f(&mut session)
    }
}

/// Build a session: static ONNX Runtime, `Level3` graph optimization,
/// `inter_op = 1`, `jobs` into the intra-op pool, the configured execution
/// provider. An on-disk model is committed from its path; a bundled model
/// from its bytes in memory.
pub(crate) fn build_session(
    source: &ModelSource,
    jobs: usize,
    config: SessionConfig,
) -> Result<Session, Error> {
    let mut builder = Session::builder().map_err(ort_error)?;
    builder = builder
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(ort_error)?;
    builder = builder.with_inter_threads(1).map_err(ort_error)?;
    if jobs > 0 {
        builder = builder
            .with_intra_threads(intra_op_threads(jobs))
            .map_err(ort_error)?;
    }
    builder = with_runner(builder, source, config)?;
    let session = match source {
        ModelSource::File(path) => builder
            .commit_from_file(path)
            .map_err(|error| Error::Ort(format!("cannot load {}: {error}", path.display())))?,
        ModelSource::Bundled { file_name, bytes } => builder
            .commit_from_memory(bytes)
            .map_err(|error| Error::Ort(format!("cannot load bundled {file_name}: {error}")))?,
    };
    if session.inputs().is_empty() || session.outputs().is_empty() {
        return Err(Error::Ort(format!(
            "{} has no input or output tensor",
            source.file_name()
        )));
    }
    Ok(session)
}

/// Register the execution provider [`SessionConfig`] selects. `Cpu` registers
/// nothing: ONNX Runtime's implicit CPU execution provider is the default.
fn with_runner(
    builder: SessionBuilder,
    #[cfg_attr(
        not(all(target_os = "macos", target_arch = "aarch64")),
        expect(unused_variables, reason = "only Core ML keys its cache by the model")
    )]
    source: &ModelSource,
    config: SessionConfig,
) -> Result<SessionBuilder, Error> {
    match config.runner {
        Runner::Cpu => Ok(builder),
        Runner::DirectMl => {
            #[cfg(all(windows, feature = "ort"))]
            {
                use ort::ep::directml::{DirectML, PerformancePreference};

                let ep = match config.directml_device {
                    Some(index) => {
                        let index = i32::try_from(index).map_err(|_| {
                            Error::Ort(format!("the DirectML device index {index} is out of range"))
                        })?;
                        DirectML::default().with_device_id(index)
                    }
                    // No explicit adapter: ONNX Runtime's own selection,
                    // preferring the high-performance GPU (device 0 is often
                    // the iGPU on laptops).
                    None => DirectML::default()
                        .with_performance_preference(PerformancePreference::HighPerformance),
                };
                builder
                    .with_execution_providers([ep.build()])
                    .map_err(|error| {
                        Error::Ort(format!(
                            "cannot register the DirectML execution provider: {error}"
                        ))
                    })
            }
            #[cfg(not(all(windows, feature = "ort")))]
            {
                Err(Error::Ort(
                    "DirectML is only available in Windows ML builds; select the CPU runner"
                        .to_string(),
                ))
            }
        }
        Runner::CoreMl => {
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            {
                builder
                    .with_execution_providers([crate::coreml::execution_provider(source)?])
                    .map_err(|error| {
                        Error::Ort(format!(
                            "cannot register the Core ML execution provider: {error}"
                        ))
                    })
            }
            #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
            {
                Err(Error::Ort(
                    "Core ML is only available in Apple-silicon macOS ML builds; select the CPU \
                     runner"
                        .to_string(),
                ))
            }
        }
    }
}

/// `jobs` clamped to something the machine can use: at least one thread, at
/// most the available parallelism (an ORT default when `jobs` is 0).
fn intra_op_threads(jobs: usize) -> usize {
    let available = std::thread::available_parallelism()
        .map(|cores| cores.get())
        .unwrap_or(1);
    jobs.clamp(1, available.max(1))
}

pub(crate) fn ort_error(error: impl std::fmt::Display) -> Error {
    Error::Ort(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intra_op_threads_clamp_to_the_machine() {
        let available = std::thread::available_parallelism()
            .map(|cores| cores.get())
            .unwrap_or(1);
        assert_eq!(intra_op_threads(1), 1);
        assert_eq!(intra_op_threads(available), available);
        assert_eq!(intra_op_threads(usize::MAX), available);
    }
}
