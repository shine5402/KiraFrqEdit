//! The process-wide ONNX Runtime session shared by every model path
//! (#48/#69): one session, built lazily behind a mutex and rebuilt only when
//! the model source or thread count changes.

use std::sync::{Mutex, MutexGuard};

use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;

use crate::Error;
use crate::model::ModelSource;

struct Cached {
    source: ModelSource,
    jobs: usize,
    session: Session,
}

static SESSION: Mutex<Option<Cached>> = Mutex::new(None);

/// Run `f` against the session for `(source, jobs)`, building or rebuilding it
/// first when needed.
pub(crate) fn with<R>(
    source: &ModelSource,
    jobs: usize,
    f: impl FnOnce(&mut Session) -> Result<R, Error>,
) -> Result<R, Error> {
    let mut guard: MutexGuard<'_, Option<Cached>> = SESSION
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let stale = guard
        .as_ref()
        .is_none_or(|cached| &cached.source != source || cached.jobs != jobs);
    if stale {
        *guard = Some(build(source, jobs)?);
    }
    let cached = guard.as_mut().expect("just built");
    f(&mut cached.session)
}

/// Build the session: static ONNX Runtime, `Level3` graph optimization,
/// `inter_op = 1`, `jobs` into the intra-op pool. An on-disk model is
/// committed from its path; a bundled model from its bytes in memory.
fn build(source: &ModelSource, jobs: usize) -> Result<Cached, Error> {
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
    Ok(Cached {
        source: source.clone(),
        jobs,
        session,
    })
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
