//! Sidecar naming and existence checks (#12).
//!
//! frq has two documented spellings — the canonical `<stem>_wav.frq` written
//! by the pipeline and the UTAU/OpenUtau alternate `<filename>.frq`, which
//! counts as an existing table so the pipeline never creates a second one.
//! pmk has no alternate spelling, and mrq lives in the folder's `desc.mrq`
//! (checked by the run through kira-frq-core's entry keys).

use std::path::{Path, PathBuf};

/// `<stem>_wav.<extension>` — e.g. `A2.wav` → `A2_wav.frq`.
pub(crate) fn canonical_table_path(wav: &Path, extension: &str) -> PathBuf {
    let mut name = wav.file_stem().unwrap_or_default().to_os_string();
    name.push("_wav.");
    name.push(extension);
    wav.with_file_name(name)
}

/// `<filename>.frq` — e.g. `A2.wav` → `A2.wav.frq`.
pub(crate) fn alternate_frq_path(wav: &Path) -> PathBuf {
    let mut name = wav.file_name().unwrap_or_default().to_os_string();
    name.push(".frq");
    wav.with_file_name(name)
}

/// A frq table under either spelling counts as existing (#12).
pub(crate) fn frq_exists(wav: &Path) -> bool {
    canonical_table_path(wav, "frq").exists() || alternate_frq_path(wav).exists()
}

/// Only the canonical pmk spelling is known.
pub(crate) fn pmk_exists(wav: &Path) -> bool {
    canonical_table_path(wav, "pmk").exists()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_replaces_the_extension_with_wav_plus_target() {
        let wav = Path::new("/bank/A2/_ka.wav");
        assert_eq!(
            canonical_table_path(wav, "frq"),
            Path::new("/bank/A2/_ka_wav.frq")
        );
        assert_eq!(
            canonical_table_path(wav, "pmk"),
            Path::new("/bank/A2/_ka_wav.pmk")
        );
        assert_eq!(
            canonical_table_path(Path::new("A2.WAV"), "frq"),
            Path::new("A2_wav.frq")
        );
    }

    #[test]
    fn alternate_appends_to_the_full_filename() {
        assert_eq!(
            alternate_frq_path(Path::new("/bank/A2/_ka.wav")),
            Path::new("/bank/A2/_ka.wav.frq")
        );
    }
}
