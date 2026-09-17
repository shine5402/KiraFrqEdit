//! moresampler `.llsm` cache invalidation (#4/#10).
//!
//! moresampler keeps one analysis cache per wav next to it, named by appending
//! `.llsm` to the full filename (`_かきくけこ.wav` → `_かきくけこ.wav.llsm`). A
//! cache fresher than the wav silently wins over `desc.mrq`, so after
//! KiraFrqGen writes a new mrq f0 entry for a wav the cache must go, or a
//! render keeps singing the old contour (`docs/research/moresampler-llsm.md`).
//! `frq` is not moresampler's f0 source, so an frq write never invalidates the
//! cache.
//!
//! Only wavs whose entry was actually written are touched — skipped and failed
//! wavs keep their caches (`no write ⇒ no delete`) — and this is the only cache
//! invalidation KiraFrqGen performs: never touch the wav, never delete
//! `desc.mrq` itself.

use std::ffi::OsString;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// The `.llsm` cache path moresampler uses for `wav`: the full filename plus
/// `.llsm`.
pub fn cache_path(wav: &Path) -> PathBuf {
    let mut cache = OsString::from(wav.as_os_str());
    cache.push(".llsm");
    PathBuf::from(cache)
}

/// Delete `wav`'s `.llsm` cache; `Ok(true)` when one was removed, `Ok(false)`
/// when there was none.
pub fn delete_cache(wav: &Path) -> io::Result<bool> {
    match fs::remove_file(cache_path(wav)) {
        Ok(()) => Ok(true),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsStr;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Scratch(PathBuf);

    impl Scratch {
        fn new(tag: &str) -> Self {
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            let path = std::env::temp_dir().join(format!(
                "frqgen-llsm-{tag}-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }

        fn join(&self, name: &str) -> PathBuf {
            self.0.join(name)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn cache_path_appends_to_the_full_filename() {
        let wav = Path::new("/bank/A2/_かきくけこ.wav");
        let cache = cache_path(wav);
        assert_eq!(cache.file_name(), Some(OsStr::new("_かきくけこ.wav.llsm")));
    }

    #[test]
    fn deletes_only_the_matching_cache() {
        let scratch = Scratch::new("delete");
        let wav = scratch.join("_かきくけこ.wav");
        fs::write(&wav, b"wav").unwrap();

        assert!(!delete_cache(&wav).unwrap(), "nothing to delete yet");

        let cache = cache_path(&wav);
        fs::write(&cache, b"cache").unwrap();
        assert!(delete_cache(&wav).unwrap());
        assert!(!cache.exists());
        assert!(wav.exists(), "the wav is never touched");
        assert!(!delete_cache(&wav).unwrap(), "already gone");
    }
}
