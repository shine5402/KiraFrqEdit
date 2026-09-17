//! The recursive `.wav` scan (#11/#21).
//!
//! The root is either a single `.wav` (taken as given) or a folder, walked
//! recursively: `wav` matches case-insensitively, macOS AppleDouble sidecars
//! (`._*`) are skipped — a bank copied from macOS otherwise doubles its wav
//! count — and symlinks and junctions are followed, so a bank may link folders
//! or files into place. A link back into a directory on the current walk is
//! cut, so a loop cannot recurse forever; a dangling link is not a wav. The
//! result is sorted by path, so a run's order is deterministic.

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
    // `metadata` follows symlinks, so a symlinked root works.
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
    let mut chain = Vec::new();
    collect(root, &mut wavs, &mut chain)?;
    wavs.sort();
    if wavs.is_empty() {
        return Err(GeneratorError::NoWavs {
            root: root.to_path_buf(),
        });
    }
    Ok(wavs)
}

/// Walk `dir` for wavs. Entries are visited in name order and `chain` holds the
/// canonical path of every directory on the current walk, so a link back into
/// an ancestor is cut rather than followed; a link into a directory elsewhere
/// in the tree is followed like a real entry.
fn collect(
    dir: &Path,
    wavs: &mut Vec<PathBuf>,
    chain: &mut Vec<PathBuf>,
) -> Result<(), GeneratorError> {
    let canonical = fs::canonicalize(dir).map_err(|error| scan_error(dir, error))?;
    if chain.contains(&canonical) {
        return Ok(());
    }
    chain.push(canonical);

    let mut entries = Vec::new();
    for entry in fs::read_dir(dir).map_err(|error| scan_error(dir, error))? {
        entries.push(entry.map_err(|error| scan_error(dir, error))?);
    }
    entries.sort_by_key(fs::DirEntry::file_name);

    for entry in entries {
        if is_appledouble(&entry.file_name()) {
            continue;
        }
        let path = entry.path();
        let entry_type = entry
            .file_type()
            .map_err(|error| scan_error(&path, error))?;
        let (is_dir, is_file) = if entry_type.is_symlink() {
            match fs::metadata(&path) {
                Ok(target) => (target.is_dir(), target.is_file()),
                Err(_) => continue, // dangling link
            }
        } else {
            (entry_type.is_dir(), entry_type.is_file())
        };
        if is_dir {
            collect(&path, wavs, chain)?;
        } else if is_file && has_wav_extension(&path) {
            wavs.push(path);
        }
    }

    chain.pop();
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
                "kirafrqgen-core-scan-{tag}-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }

        /// A path inside the scratch folder; nothing is created.
        fn path(&self, name: &str) -> PathBuf {
            self.0.join(name)
        }

        /// Write a placeholder file, creating parent folders.
        fn file(&self, name: &str) -> PathBuf {
            let path = self.path(name);
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
        scratch.file("B2.wav");
        scratch.file("A2/A2.wav");
        scratch.file("A2/A3.WAV");
        scratch.file("A2/notes.txt");
        scratch.file("A2/A3.wave");
        scratch.file("A2/._A4.wav");

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
        scratch.file("._junk/inside.wav");
        scratch.file("real.wav");

        let wavs = scan_wavs(&scratch.0).unwrap();
        assert_eq!(names(&wavs, &scratch), ["real.wav"]);
    }

    #[test]
    fn accepts_a_single_wav_root_case_insensitively() {
        let scratch = Scratch::new("single");
        let wav = scratch.file("A2.WAV");

        assert_eq!(scan_wavs(&wav).unwrap(), [wav]);
    }

    #[test]
    fn rejects_an_appledouble_file_root() {
        let scratch = Scratch::new("appledouble-root");
        let file = scratch.file("._A2.wav");

        assert!(matches!(scan_wavs(&file), Err(GeneratorError::Config(_))));
    }

    #[test]
    fn rejects_a_non_wav_file_root() {
        let scratch = Scratch::new("nonwav");
        let file = scratch.file("notes.txt");

        assert!(matches!(scan_wavs(&file), Err(GeneratorError::Config(_))));
    }

    #[test]
    fn empty_folder_and_missing_root_are_fatal() {
        let scratch = Scratch::new("empty");
        assert!(matches!(
            scan_wavs(&scratch.0),
            Err(GeneratorError::NoWavs { .. })
        ));

        let missing = scratch.path("missing");
        assert!(matches!(
            scan_wavs(&missing),
            Err(GeneratorError::Scan { .. })
        ));
    }

    #[cfg(unix)]
    #[test]
    fn symlinks_to_folders_and_files_are_followed() {
        let scratch = Scratch::new("symlink");
        scratch.file("real/inside.wav");
        std::os::unix::fs::symlink(scratch.path("real"), scratch.path("linked")).unwrap();
        std::os::unix::fs::symlink(scratch.path("real/inside.wav"), scratch.path("alias.wav"))
            .unwrap();

        let wavs = scan_wavs(&scratch.0).unwrap();
        assert_eq!(
            names(&wavs, &scratch),
            ["alias.wav", "linked/inside.wav", "real/inside.wav"]
        );
    }

    #[cfg(unix)]
    #[test]
    fn symlink_cycles_terminate() {
        let scratch = Scratch::new("symlink-loop");
        scratch.file("real/inside.wav");
        std::os::unix::fs::symlink(&scratch.0, scratch.path("real/loop")).unwrap();
        std::os::unix::fs::symlink(scratch.path("real"), scratch.path("real/self")).unwrap();

        let wavs = scan_wavs(&scratch.0).unwrap();
        assert_eq!(names(&wavs, &scratch), ["real/inside.wav"]);
    }

    #[cfg(unix)]
    #[test]
    fn dangling_symlinks_are_skipped() {
        let scratch = Scratch::new("dangling");
        std::os::unix::fs::symlink(scratch.path("missing"), scratch.path("gone.wav")).unwrap();
        scratch.file("real.wav");

        let wavs = scan_wavs(&scratch.0).unwrap();
        assert_eq!(names(&wavs, &scratch), ["real.wav"]);
    }
}
