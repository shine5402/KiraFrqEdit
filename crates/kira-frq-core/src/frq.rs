//! UTAU classic `frq`: f0 + amplitude + key frequency.
//!
//! Format spec: `docs/research/frq-format.md`; the writer policy (amplitude
//! scale, key frequency, frame counts) was settled in #8. The module converts
//! to and from the neutral [`FrequencyTable`] (ADR 0001) and is path-agnostic:
//! it reads and writes whatever path the pipeline hands it, sidecar naming
//! included.
//!
//! Reading makes the header's frame count authoritative: exactly that many
//! pairs are read, a file shorter than the count declares is rejected as
//! truncated, and extra bytes after the declared frames (appended data) are
//! ignored. The 16 reserved bytes are ignored too (real files store a sample
//! rate or a SpeedWagon marker there).
//!
//! Writing zeroes those reserved bytes. The amplitude comes from
//! [`FrequencyTable::amplitude`] (the pipeline computes it per #8); frames it
//! does not cover are written as `0.0`.

use std::io;
use std::path::Path;

use crate::FrequencyTable;

/// The exact magic every file starts with.
pub const MAGIC: &[u8; 8] = b"FREQ0003";

/// `8` magic + `4` hop + `8` key + `16` reserved + `4` frame count.
pub const HEADER_LEN: usize = 40;

/// One frame: an `f64` f0 immediately followed by an `f64` amplitude.
pub const FRAME_LEN: usize = 16;

/// `frq` has no sample-rate field; UTAU assumes 44.1 kHz.
pub const SAMPLE_RATE: u32 = 44_100;

/// Parse a complete file image into the neutral table.
///
/// The header's frame count decides how many frames are read; extra bytes
/// after them are ignored (see the module docs). f0 and the key frequency are
/// preserved unrounded and the amplitude always lands in
/// [`FrequencyTable::amplitude`].
pub fn parse(bytes: &[u8]) -> Result<FrequencyTable, FrqError> {
    let header = bytes.get(..HEADER_LEN).ok_or(FrqError::Truncated)?;
    if header[..8] != MAGIC[..] {
        return Err(FrqError::BadMagic);
    }
    let hop = i32::from_le_bytes(header[8..12].try_into().expect("four bytes"));
    if hop <= 0 {
        return Err(FrqError::InvalidHop { hop });
    }
    let key_hz = f64::from_le_bytes(header[12..20].try_into().expect("eight bytes"));
    let count = i32::from_le_bytes(header[36..40].try_into().expect("four bytes"));
    if count < 0 {
        return Err(FrqError::InvalidCount { count });
    }
    let count = count as usize;

    let available = (bytes.len() - HEADER_LEN) / FRAME_LEN;
    if count > available {
        return Err(FrqError::Truncated);
    }
    let frames = &bytes[HEADER_LEN..HEADER_LEN + count * FRAME_LEN];

    let mut f0_hz = Vec::with_capacity(count);
    let mut amplitude = Vec::with_capacity(count);
    for pair in frames.as_chunks::<FRAME_LEN>().0 {
        f0_hz.push(f64::from_le_bytes(
            pair[..8].try_into().expect("eight bytes"),
        ));
        amplitude.push(f64::from_le_bytes(
            pair[8..].try_into().expect("eight bytes"),
        ));
    }

    Ok(FrequencyTable {
        sample_rate: SAMPLE_RATE,
        hop_samples: hop as u32,
        f0_hz,
        amplitude: Some(amplitude),
        key_hz,
    })
}

/// Read and parse `path`.
pub fn read(path: &Path) -> Result<FrequencyTable, ReadError> {
    let bytes = std::fs::read(path).map_err(ReadError::Io)?;
    parse(&bytes).map_err(ReadError::Parse)
}

/// Serialize `table` to a complete file image.
///
/// The reserved bytes are zeroed (see the module docs).
pub fn to_bytes(table: &FrequencyTable) -> Vec<u8> {
    debug_assert!(
        table.hop_samples <= i32::MAX as u32,
        "hop must fit the header's int32"
    );
    debug_assert!(
        table.f0_hz.len() <= i32::MAX as usize,
        "the frame count must fit the header's int32"
    );
    debug_assert!(
        table
            .amplitude
            .as_ref()
            .is_none_or(|amplitudes| amplitudes.len() == table.f0_hz.len()),
        "amplitude must cover every frame when present"
    );

    let mut out = Vec::with_capacity(HEADER_LEN + FRAME_LEN * table.f0_hz.len());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&(table.hop_samples as i32).to_le_bytes());
    out.extend_from_slice(&table.key_hz.to_le_bytes());
    out.extend_from_slice(&[0u8; 16]);
    out.extend_from_slice(&(table.f0_hz.len() as i32).to_le_bytes());
    for (index, &f0) in table.f0_hz.iter().enumerate() {
        let amplitude = table
            .amplitude
            .as_ref()
            .and_then(|amplitudes| amplitudes.get(index))
            .copied()
            .unwrap_or(0.0);
        out.extend_from_slice(&f0.to_le_bytes());
        out.extend_from_slice(&amplitude.to_le_bytes());
    }
    out
}

/// Write `table` to `path` as a complete file image.
pub fn write(path: &Path, table: &FrequencyTable) -> io::Result<()> {
    std::fs::write(path, to_bytes(table))
}

/// Everything about a byte image that makes it not a `frq` table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FrqError {
    /// The magic is not `FREQ0003` (no `FREQ0001`/`FREQ0002` is known).
    BadMagic,
    /// The image is shorter than the 40-byte header, or shorter than the
    /// header's frame count declares.
    Truncated,
    /// The header frame count is negative.
    InvalidCount { count: i32 },
    /// The header `hop` is not a positive sample count.
    InvalidHop { hop: i32 },
}

impl std::fmt::Display for FrqError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FrqError::BadMagic => write!(f, "not a frq file (bad magic)"),
            FrqError::Truncated => write!(f, "truncated frq data"),
            FrqError::InvalidCount { count } => {
                write!(f, "frq header has invalid frame count {count}")
            }
            FrqError::InvalidHop { hop } => write!(f, "frq header has invalid hop {hop}"),
        }
    }
}

impl std::error::Error for FrqError {}

/// [`read`] can fail either reading or parsing.
pub type ReadError = crate::ReadError<FrqError>;
