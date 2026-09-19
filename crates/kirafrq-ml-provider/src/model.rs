//! Model-file resolution (#48): user-supplied weights only, nothing hosted or
//! redistributed by us. The lookup is the executable's directory, then the
//! `KIRAFRQ_ML_DIR` environment variable; the expected filename is
//! `rmvpe.onnx`; existence is availability.

use std::path::{Path, PathBuf};

/// The expected model filename in every lookup location.
pub const MODEL_FILE_NAME: &str = "rmvpe.onnx";

/// Environment variable holding an extra directory to look in, after the
/// executable's own directory.
pub const MODEL_DIR_ENV: &str = "KIRAFRQ_ML_DIR";

/// Why a model file is unavailable; the message names the file, the lookup
/// directories and the environment variable, so the CLI/GUI hint is concrete.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelError {
    pub file_name: String,
    pub searched: Vec<PathBuf>,
}

impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let dirs = self
            .searched
            .iter()
            .map(|dir| dir.display().to_string())
            .collect::<Vec<_>>()
            .join(", ");
        write!(
            f,
            "{} was not found; put it next to the executable or in ${MODEL_DIR_ENV} \
             (searched: {dirs})",
            self.file_name
        )
    }
}

impl std::error::Error for ModelError {}

/// Resolve the model file: `<exe dir>/rmvpe.onnx`, then
/// `$KIRAFRQ_ML_DIR/rmvpe.onnx`.
pub fn resolve_model() -> Result<PathBuf, ModelError> {
    resolve_model_in(
        executable_dir(),
        std::env::var_os(MODEL_DIR_ENV)
            .map(PathBuf::from)
            .filter(|dir| !dir.as_os_str().is_empty()),
    )
}

/// The lookup itself, over explicit directories so it is testable without
/// touching the process environment.
fn resolve_model_in(
    executable_dir: Option<PathBuf>,
    env_dir: Option<PathBuf>,
) -> Result<PathBuf, ModelError> {
    let mut searched = Vec::new();
    for dir in [executable_dir, env_dir].into_iter().flatten() {
        let candidate = dir.join(MODEL_FILE_NAME);
        if candidate.is_file() {
            return Ok(candidate);
        }
        searched.push(dir);
    }
    Err(ModelError {
        file_name: MODEL_FILE_NAME.to_string(),
        searched,
    })
}

/// The executable's directory, if the platform reports one.
fn executable_dir() -> Option<PathBuf> {
    std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(Path::to_path_buf))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_file_names_itself_the_dirs_and_the_env_var() {
        let error = ModelError {
            file_name: MODEL_FILE_NAME.to_string(),
            searched: vec![PathBuf::from("A"), PathBuf::from("B")],
        };
        let message = error.to_string();
        assert!(message.contains("rmvpe.onnx"), "{message}");
        assert!(message.contains("KIRAFRQ_ML_DIR"), "{message}");
        assert!(message.contains("A"), "{message}");
        assert!(message.contains("B"), "{message}");
    }

    #[test]
    fn the_lookup_prefers_the_executable_directory_then_the_env_dir() {
        let scratch =
            std::env::temp_dir().join(format!("kirafrq-ml-provider-lookup-{}", std::process::id()));
        let exe_dir = scratch.join("exe");
        let env_dir = scratch.join("env");
        std::fs::create_dir_all(&exe_dir).unwrap();
        std::fs::create_dir_all(&env_dir).unwrap();

        // Neither directory has the file: both are reported as searched.
        let error = resolve_model_in(Some(exe_dir.clone()), Some(env_dir.clone())).unwrap_err();
        assert_eq!(error.searched, vec![exe_dir.clone(), env_dir.clone()]);

        // Only the env dir: the file resolves there.
        std::fs::write(env_dir.join(MODEL_FILE_NAME), b"model").unwrap();
        assert_eq!(
            resolve_model_in(Some(exe_dir.clone()), Some(env_dir.clone())).unwrap(),
            env_dir.join(MODEL_FILE_NAME)
        );

        // Both: the executable directory wins.
        std::fs::write(exe_dir.join(MODEL_FILE_NAME), b"model").unwrap();
        assert_eq!(
            resolve_model_in(Some(exe_dir.clone()), Some(env_dir)).unwrap(),
            exe_dir.join(MODEL_FILE_NAME)
        );

        // No directories at all is still an error, not a panic.
        assert!(resolve_model_in(None, None).is_err());
        let _ = std::fs::remove_dir_all(&scratch);
    }

    #[test]
    fn a_lookup_with_no_directories_reports_nothing_searched() {
        let error = resolve_model_in(None, None).unwrap_err();
        assert!(error.searched.is_empty(), "{:?}", error.searched);
    }
}
