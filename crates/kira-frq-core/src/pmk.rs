//! TIPS `pmk`: f0 as a period-segment walk of codes.
//!
//! Format spec: `docs/research/pmk-format.md`; the encoding rules were settled
//! in #9.
//!
//! The writer projects a [`FrequencyTable`] onto the 44.1 kHz sample timeline
//! as `(pos_end, code)` period segments: `code` is the nearest integer period
//! in samples, `49` marks unvoiced, and the walk advances by the *unrounded*
//! period so the rounding never accumulates into drift. f0 outside TIPS's
//! detected range (86–882 Hz, i.e. codes `50..=513`) is stored as unvoiced,
//! since the writer does not invent a range TIPS does not have.
//!
//! The module is path-agnostic: the pipeline owns sidecar naming
//! (`<stem>_wav.pmk`, #9) and passes the normalized wav length, because the
//! table's frame count does not determine `L` exactly.

use std::fs;
use std::io;
use std::path::Path;

use crate::FrequencyTable;

/// Header version written by the writer (TIPS 0.19β).
pub const VERSION: i16 = 19;

/// The `code` that marks an unvoiced segment.
pub const UNVOICED_CODE: i32 = 49;

/// Lowest voiced `code` (44100/50 = 882 Hz).
pub const MIN_VOICED_CODE: i32 = 50;

/// Highest voiced `code` (44100/513 ≈ 86 Hz).
pub const MAX_VOICED_CODE: i32 = 513;

/// The sample rate the pmk timeline is defined on.
pub const SAMPLE_RATE: u32 = 44_100;

/// The fixed step an unvoiced segment advances (#9).
const UNVOICED_PERIOD: f64 = UNVOICED_CODE as f64;

/// The unrounded period of a voiced frame, in 44.1 kHz samples.
fn voiced_period(f0_hz: f64) -> f64 {
    SAMPLE_RATE as f64 / f0_hz
}

/// The stored code and walk period for one frame: `49` and a 49-sample step for
/// a zero, non-finite or out-of-range f0, else the rounded code and the
/// unrounded period the walk advances by.
fn frame_step(f0_hz: f64) -> (i32, f64) {
    if !f0_hz.is_finite() || f0_hz <= 0.0 {
        return (UNVOICED_CODE, UNVOICED_PERIOD);
    }
    let period = voiced_period(f0_hz);
    let code = period.round() as i32;
    if (MIN_VOICED_CODE..=MAX_VOICED_CODE).contains(&code) {
        (code, period)
    } else {
        (UNVOICED_CODE, UNVOICED_PERIOD)
    }
}

/// The f0 of the table frame containing sample position `pos` (floor lookup on
/// the table's hop). A table with no frames reads as unvoiced; positions past
/// the last frame read it.
fn frame_at(table: &FrequencyTable, pos: f64) -> f64 {
    if table.f0_hz.is_empty() {
        return 0.0;
    }
    let hop = f64::from(table.hop_samples.max(1));
    let index = ((pos / hop).floor() as usize).min(table.f0_hz.len() - 1);
    table.f0_hz[index]
}

/// The #9 period walk: advance by the unrounded period, emit `(round(pos),
/// code)`. The emitted mark is rounded, so the stop is too: `round(pos +
/// period) >= L` aborts even when the unrounded position is still below `L`,
/// which keeps every `pos_end` below `L` (contract 3 of #9).
fn walk(table: &FrequencyTable, length_samples: usize) -> Vec<(i32, i32)> {
    let length = length_samples as f64;
    let mut entries = Vec::new();
    let mut pos = 0.0f64;
    loop {
        let (code, period) = frame_step(frame_at(table, pos));
        let next = pos + period;
        if next.round() >= length {
            return entries;
        }
        pos = next;
        entries.push((pos.round() as i32, code));
    }
}

/// Encode `table` for a wav of `length_samples` (after normalization) into a
/// complete pmk file: the 14-byte header plus the #9 period walk.
///
/// `avg` is the f64 mean of the emitted voiced codes (one vote per entry),
/// `0.0` when the file has none — including the header-only file a wav of
/// `<= 49` samples produces.
pub fn to_bytes(table: &FrequencyTable, length_samples: usize) -> Vec<u8> {
    debug_assert_eq!(
        table.sample_rate, SAMPLE_RATE,
        "pmk is defined on the 44.1 kHz timeline; normalize before writing (#11)"
    );
    let entries = walk(table, length_samples);

    let mut voiced_sum = 0.0f64;
    let mut voiced_count = 0u64;
    for &(_, code) in &entries {
        if code != UNVOICED_CODE {
            voiced_sum += f64::from(code);
            voiced_count += 1;
        }
    }
    let avg = if voiced_count == 0 {
        0.0
    } else {
        voiced_sum / voiced_count as f64
    };

    let mut out = Vec::with_capacity(14 + 8 * entries.len());
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&avg.to_le_bytes());
    out.extend_from_slice(&(entries.len() as i32).to_le_bytes());
    for (pos_end, code) in entries {
        out.extend_from_slice(&pos_end.to_le_bytes());
        out.extend_from_slice(&code.to_le_bytes());
    }
    out
}

pub fn write(path: &Path, table: &FrequencyTable, length_samples: usize) -> io::Result<()> {
    fs::write(path, to_bytes(table, length_samples))
}
