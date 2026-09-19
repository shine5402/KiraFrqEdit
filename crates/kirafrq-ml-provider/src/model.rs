//! Model resolution (#48/#69): the lookup is the executable's directory,
//! then the `KIRAFRQ_ML_DIR` environment variable.
//!
//! RMVPE is user-supplied weights only, nothing hosted or redistributed by us.
//! SwiftF0 ships as a bundled model (MIT); the on-disk lookup is its override.

use std::path::{Path, PathBuf};

/// RMVPE's expected filename in every lookup location.
pub const RMVPE_FILE_NAME: &str = "rmvpe.onnx";

/// SwiftF0's expected filename: the on-disk override for the bundled model.
pub const SWIFTF0_FILE_NAME: &str = "swiftf0.onnx";

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

/// Where an estimator's weights come from: a resolved on-disk file (the
/// user's override) or bytes embedded in the binary (SwiftF0's bundled model).
#[derive(Debug, Clone)]
pub enum ModelSource {
    File(PathBuf),
    Bundled {
        file_name: &'static str,
        bytes: &'static [u8],
    },
}

impl ModelSource {
    /// The model's filename, for error messages and session identity.
    pub fn file_name(&self) -> &str {
        match self {
            ModelSource::File(path) => path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("model.onnx"),
            ModelSource::Bundled { file_name, .. } => file_name,
        }
    }
}

/// Two bundled sources are equal when they carry the same filename (there is
/// one bundled model per name); comparing the 398 KB byte slices on every
/// estimate would be wasteful.
impl PartialEq for ModelSource {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (ModelSource::File(left), ModelSource::File(right)) => left == right,
            (
                ModelSource::Bundled {
                    file_name: left, ..
                },
                ModelSource::Bundled {
                    file_name: right, ..
                },
            ) => left == right,
            _ => false,
        }
    }
}

impl Eq for ModelSource {}

impl From<PathBuf> for ModelSource {
    fn from(path: PathBuf) -> Self {
        ModelSource::File(path)
    }
}

impl From<&Path> for ModelSource {
    fn from(path: &Path) -> Self {
        ModelSource::File(path.to_path_buf())
    }
}

impl From<String> for ModelSource {
    fn from(path: String) -> Self {
        ModelSource::File(PathBuf::from(path))
    }
}

impl From<&str> for ModelSource {
    fn from(path: &str) -> Self {
        ModelSource::File(PathBuf::from(path))
    }
}

/// Resolve RMVPE's model file: `<exe dir>/rmvpe.onnx`, then
/// `$KIRAFRQ_ML_DIR/rmvpe.onnx`.
pub fn resolve_model() -> Result<PathBuf, ModelError> {
    resolve_file(RMVPE_FILE_NAME)
}

/// Resolve `file_name` in the executable's directory, then
/// `$KIRAFRQ_ML_DIR`.
pub fn resolve_file(file_name: &str) -> Result<PathBuf, ModelError> {
    resolve_file_in(
        file_name,
        executable_dir(),
        std::env::var_os(MODEL_DIR_ENV)
            .map(PathBuf::from)
            .filter(|dir| !dir.as_os_str().is_empty()),
    )
}

/// The lookup itself, over explicit directories so it is testable without
/// touching the process environment.
fn resolve_file_in(
    file_name: &str,
    executable_dir: Option<PathBuf>,
    env_dir: Option<PathBuf>,
) -> Result<PathBuf, ModelError> {
    let mut searched = Vec::new();
    for dir in [executable_dir, env_dir].into_iter().flatten() {
        let candidate = dir.join(file_name);
        if candidate.is_file() {
            return Ok(candidate);
        }
        searched.push(dir);
    }
    Err(ModelError {
        file_name: file_name.to_string(),
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
            file_name: RMVPE_FILE_NAME.to_string(),
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
        let error = resolve_file_in(
            RMVPE_FILE_NAME,
            Some(exe_dir.clone()),
            Some(env_dir.clone()),
        )
        .unwrap_err();
        assert_eq!(error.searched, vec![exe_dir.clone(), env_dir.clone()]);

        // Only the env dir: the file resolves there.
        std::fs::write(env_dir.join(RMVPE_FILE_NAME), b"model").unwrap();
        assert_eq!(
            resolve_file_in(
                RMVPE_FILE_NAME,
                Some(exe_dir.clone()),
                Some(env_dir.clone())
            )
            .unwrap(),
            env_dir.join(RMVPE_FILE_NAME)
        );

        // Both: the executable directory wins.
        std::fs::write(exe_dir.join(RMVPE_FILE_NAME), b"model").unwrap();
        assert_eq!(
            resolve_file_in(RMVPE_FILE_NAME, Some(exe_dir.clone()), Some(env_dir)).unwrap(),
            exe_dir.join(RMVPE_FILE_NAME)
        );

        // No directories at all is still an error, not a panic.
        assert!(resolve_file_in(RMVPE_FILE_NAME, None, None).is_err());
        let _ = std::fs::remove_dir_all(&scratch);
    }

    #[test]
    fn the_lookup_reports_the_asked_file_name() {
        // The same lookup is what SwiftF0's on-disk override uses, so it must
        // report the name it was given.
        let error = resolve_file_in(SWIFTF0_FILE_NAME, None, None).unwrap_err();
        assert_eq!(error.file_name, "swiftf0.onnx");
        assert!(error.searched.is_empty(), "{:?}", error.searched);
    }

    #[test]
    fn model_sources_compare_by_identity_not_bytes() {
        let file = ModelSource::File(PathBuf::from("a/swiftf0.onnx"));
        assert_eq!(file, ModelSource::from(PathBuf::from("a/swiftf0.onnx")));
        assert_eq!(file.file_name(), "swiftf0.onnx");

        let bytes: &'static [u8] = b"weights";
        let bundled = ModelSource::Bundled {
            file_name: SWIFTF0_FILE_NAME,
            bytes,
        };
        assert_eq!(
            bundled,
            ModelSource::Bundled {
                file_name: SWIFTF0_FILE_NAME,
                bytes
            }
        );
        assert_ne!(file, bundled, "a file is never the bundled model");
    }
}
