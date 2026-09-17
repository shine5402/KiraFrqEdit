//! The recursive `.wav` scan (#11/#21).
//!
//! The root is either a single `.wav` (taken as given) or a folder, walked
//! recursively: `wav` matches case-insensitively, macOS AppleDouble sidecars
//! (`._*`) are skipped — a bank copied from macOS otherwise doubles its wav
//! count — and directory symlinks are not followed. The result is sorted by
//! path, so a run's order is deterministic.

use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};

use crate::GeneratorError;

/// Whether `name` is a macOS AppleDouble sidecar (`._*`).
fn is_appledouble(name: &OsStr) -> bool {
    name.as_encoded_bytes().starts_with(b"._")
}

/// `.wav`, case-insensitively; `.wave` is not matched (#11).
fn has_wav_extension(path: &Path) -> bool {
    path.extension()
        .is_some_and(|extension| extension.as_encoded_bytes().eq_ignore_ascii_case(b"wav"))
}

/// Every wav `root` names: the file itself, or the folder's wavs recursively.
///
/// Fatal when the root cannot be read, is neither a wav nor a folder, or when
/// a folder holds no wavs at all.
pub(crate) fn scan_wavs(root: &Path) -> Result<Vec<PathBuf>, GeneratorError> {
    // `metadata` follows symlinks, so a symlinked root works; nested
    // directory symlinks are not followed (see `collect`).
    let metadata = fs::metadata(root).map_err(|error| scan_error(root, error))?;

    if metadata.is_file() {
        if is_appledouble(root.file_name().unwrap_or_default()) {
            return Err(GeneratorError::Config(format!(
                "{} is a macOS AppleDouble sidecar",
                root.display()
            )));
        }
        return if has_wav_extension(root) {
            Ok(vec![root.to_path_buf()])
        } else {
            Err(GeneratorError::Config(format!(
                "{} is not a .wav file",
                root.display()
            )))
        };
    }
    if !metadata.is_dir() {
        return Err(GeneratorError::Config(format!(
            "{} is neither a .wav file nor a folder",
            root.display()
        )));
    }

    let mut wavs = Vec::new();
    collect(root, &mut wavs)?;
    wavs.sort();
    if wavs.is_empty() {
        return Err(GeneratorError::NoWavs {
            root: root.to_path_buf(),
        });
    }
    Ok(wavs)
}

/// Walk `dir` for wavs. `DirEntry::file_type` does not resolve symlinks, so a
/// symlinked directory reports as a symlink and is skipped, never recursed.
fn collect(dir: &Path, wavs: &mut Vec<PathBuf>) -> Result<(), GeneratorError> {
    let entries = fs::read_dir(dir).map_err(|error| scan_error(dir, error))?;
    for entry in entries {
        let entry = entry.map_err(|error| scan_error(dir, error))?;
        if is_appledouble(&entry.file_name()) {
            continue;
        }
        let path = entry.path();
        let file_type = entry
            .file_type()
            .map_err(|error| scan_error(&path, error))?;
        if file_type.is_dir() {
            collect(&path, wavs)?;
        } else if file_type.is_file() && has_wav_extension(&path) {
            wavs.push(path);
        }
    }
    Ok(())
}

fn scan_error(path: &Path, error: std::io::Error) -> GeneratorError {
    GeneratorError::Scan {
        path: path.to_path_buf(),
        error,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Scratch(PathBuf);

    impl Scratch {
        fn new(tag: &str) -> Self {
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            let path = std::env::temp_dir().join(format!(
                "kira-frqgen-scan-{tag}-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }

        fn join(&self, name: &str) -> PathBuf {
            let path = self.0.join(name);
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).unwrap();
            }
            fs::write(&path, b"wav").unwrap();
            path
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn names(wavs: &[PathBuf], scratch: &Scratch) -> Vec<String> {
        wavs.iter()
            .map(|wav| {
                wav.strip_prefix(&scratch.0)
                    .unwrap()
                    .to_string_lossy()
                    .replace('\\', "/")
            })
            .collect()
    }

    #[test]
    fn finds_nested_wavs_sorted_and_ignores_other_files() {
        let scratch = Scratch::new("nested");
        scratch.join("B2.wav");
        scratch.join("A2/A2.wav");
        scratch.join("A2/A3.WAV");
        scratch.join("A2/notes.txt");
        scratch.join("A2/A3.wave");
        scratch.join("A2/._A4.wav");

        let wavs = scan_wavs(&scratch.0).unwrap();
        assert_eq!(
            names(&wavs, &scratch),
            ["A2/A2.wav", "A2/A3.WAV", "B2.wav"],
            "sorted by path, case-insensitive .wav, no .wave, no AppleDouble"
        );
    }

    #[test]
    fn appledouble_directories_are_skipped_too() {
        let scratch = Scratch::new("appledouble");
        scratch.join("._junk/inside.wav");
        scratch.join("real.wav");

        let wavs = scan_wavs(&scratch.0).unwrap();
        assert_eq!(names(&wavs, &scratch), ["real.wav"]);
    }

    #[test]
    fn accepts_a_single_wav_root_case_insensitively() {
        let scratch = Scratch::new("single");
        let wav = scratch.join("A2.WAV");

        assert_eq!(scan_wavs(&wav).unwrap(), [wav]);
    }

    #[test]
    fn rejects_an_appledouble_file_root() {
        let scratch = Scratch::new("appledouble-root");
        let file = scratch.join("._A2.wav");

        assert!(matches!(scan_wavs(&file), Err(GeneratorError::Config(_))));
    }

    #[test]
    fn rejects_a_non_wav_file_root() {
        let scratch = Scratch::new("nonwav");
        let file = scratch.join("notes.txt");

        assert!(matches!(scan_wavs(&file), Err(GeneratorError::Config(_))));
    }

    #[test]
    fn empty_folder_and_missing_root_are_fatal() {
        let scratch = Scratch::new("empty");
        assert!(matches!(
            scan_wavs(&scratch.0),
            Err(GeneratorError::NoWavs { .. })
        ));

        let missing = scratch.join("missing");
        fs::remove_file(&missing).unwrap();
        assert!(matches!(
            scan_wavs(&missing),
            Err(GeneratorError::Scan { .. })
        ));
    }

    #[cfg(unix)]
    #[test]
    fn directory_symlinks_are_not_followed() {
        let scratch = Scratch::new("symlink");
        scratch.join("real/inside.wav");
        std::os::unix::fs::symlink(scratch.join("real"), scratch.join("linked")).unwrap();

        let wavs = scan_wavs(&scratch.0).unwrap();
        assert_eq!(names(&wavs, &scratch), ["real/inside.wav"]);
    }
}
