//! Structural contract for `kira_frq_core::frq` — the #17 checklist over
//! synthetic, patched and truncated fixtures.
//!
//! Raw byte fixtures are built and decoded independently of the module under
//! test, so the assertions pin the on-disk layout, not a round-trip echo.

use std::fs;
use std::io;

use kira_frq_core::FrequencyTable;
use kira_frq_core::frq::{self, FrqError, ReadError};

mod common;
use common::Scratch;

const MAGIC: &[u8; 8] = b"FREQ0003";
const HEADER_LEN: usize = 40;
const FRAME_LEN: usize = 16;

fn f64_bytes(value: f64) -> [u8; 8] {
    value.to_le_bytes()
}

/// One raw file: header (`n` is written verbatim, possibly disagreeing with
/// the frame count), reserved bytes, then `(f0, amp)` pairs.
fn frq_file(hop: i32, key: f64, reserved: &[u8; 16], n: i32, frames: &[(f64, f64)]) -> Vec<u8> {
    let mut out = Vec::with_capacity(HEADER_LEN + FRAME_LEN * frames.len());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&hop.to_le_bytes());
    out.extend_from_slice(&f64_bytes(key));
    out.extend_from_slice(reserved);
    out.extend_from_slice(&n.to_le_bytes());
    for &(f0, amp) in frames {
        out.extend_from_slice(&f64_bytes(f0));
        out.extend_from_slice(&f64_bytes(amp));
    }
    out
}

fn table(f0_hz: Vec<f64>, amplitude: Option<Vec<f64>>, key_hz: f64) -> FrequencyTable {
    FrequencyTable {
        sample_rate: 44_100,
        hop_samples: 256,
        f0_hz,
        amplitude,
        key_hz,
    }
}

/// Independent raw decode: `(hop, key, reserved, n, frames)`.
fn decode(bytes: &[u8]) -> (i32, f64, [u8; 16], i32, Vec<(f64, f64)>) {
    assert_eq!(&bytes[..8], MAGIC, "magic");
    let hop = i32::from_le_bytes(bytes[8..12].try_into().unwrap());
    let key = f64::from_le_bytes(bytes[12..20].try_into().unwrap());
    let mut reserved = [0u8; 16];
    reserved.copy_from_slice(&bytes[20..36]);
    let n = i32::from_le_bytes(bytes[36..40].try_into().unwrap());
    let mut frames = Vec::new();
    let mut pos = HEADER_LEN;
    while pos + FRAME_LEN <= bytes.len() {
        let f0 = f64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap());
        let amp = f64::from_le_bytes(bytes[pos + 8..pos + 16].try_into().unwrap());
        frames.push((f0, amp));
        pos += FRAME_LEN;
    }
    assert_eq!(pos, bytes.len(), "the file must be consumed exactly");
    (hop, key, reserved, n, frames)
}

/// Reading: magic checked, hop/key/count read from the header, `(f0, amp)`
/// pairs decoded in order.
#[test]
fn parse_reads_the_header_and_the_interleaved_frames() {
    let frames = [(440.0, 1234.5), (0.0, 0.0), (220.0, 88.0)];
    let bytes = frq_file(256, 440.0, &[0; 16], 3, &frames);

    let parsed = frq::parse(&bytes).unwrap();
    assert_eq!(parsed.sample_rate, 44_100, "frq assumes 44.1 kHz");
    assert_eq!(parsed.hop_samples, 256);
    assert_eq!(parsed.f0_hz, vec![440.0, 0.0, 220.0]);
    assert_eq!(parsed.amplitude, Some(vec![1234.5, 0.0, 88.0]));
    assert_eq!(parsed.key_hz, 440.0);
}

/// Reading: the header count is authoritative — exactly that many frames are
/// read, and extra bytes after them (appended data) are ignored.
#[test]
fn header_count_wins_and_extra_trailing_data_is_ignored() {
    let frames = [(100.0, 1.0), (200.0, 2.0), (0.0, 3.0)];
    for declared in [0i32, 1, 3] {
        let mut bytes = frq_file(256, 150.0, &[0; 16], declared, &frames);
        bytes.extend_from_slice(&[0xAB; 8]);

        let parsed = frq::parse(&bytes).unwrap();
        let count = declared as usize;
        assert_eq!(
            parsed.f0_hz,
            frames[..count]
                .iter()
                .map(|&(f0, _)| f0)
                .collect::<Vec<_>>(),
            "header count {declared}"
        );
        assert_eq!(
            parsed.amplitude,
            Some(
                frames[..count]
                    .iter()
                    .map(|&(_, amp)| amp)
                    .collect::<Vec<_>>()
            ),
            "header count {declared}"
        );
    }
}

/// Reading: a file with fewer frames than the header declares is truncated,
/// never partially read.
#[test]
fn a_file_shorter_than_its_header_count_is_rejected() {
    let frames = [(100.0, 1.0), (200.0, 2.0), (0.0, 3.0)];
    for declared in [4i32, 5, i32::MAX] {
        let bytes = frq_file(256, 150.0, &[0; 16], declared, &frames);
        assert_eq!(
            frq::parse(&bytes),
            Err(FrqError::Truncated),
            "header count {declared}"
        );
    }
}

/// Sanity check: a negative frame count is a malformed header, not a
/// truncation.
#[test]
fn a_negative_frame_count_is_a_typed_error() {
    let bytes = frq_file(256, 440.0, &[0; 16], -2, &[(440.0, 1.0)]);
    assert_eq!(
        frq::parse(&bytes),
        Err(FrqError::InvalidCount { count: -2 })
    );
}

/// Reading robustness 1: a non-`FREQ0003` magic is rejected as a typed error,
/// and a file shorter than the header is truncated rather than misread.
#[test]
fn wrong_magic_and_short_files_are_typed_errors() {
    let mut old_magic = frq_file(256, 440.0, &[0; 16], 0, &[]);
    old_magic[..8].copy_from_slice(b"FREQ0002");
    assert_eq!(frq::parse(&old_magic), Err(FrqError::BadMagic));
    assert_eq!(frq::parse(b"FREQ0003"), Err(FrqError::Truncated));
    assert_eq!(frq::parse(&[]), Err(FrqError::Truncated));

    let mut short = frq_file(256, 440.0, &[0; 16], 0, &[]);
    short.truncate(HEADER_LEN - 1);
    assert_eq!(frq::parse(&short), Err(FrqError::Truncated));

    let header_only = frq_file(256, 440.0, &[0; 16], 0, &[]);
    let parsed = frq::parse(&header_only).unwrap();
    assert!(parsed.f0_hz.is_empty());
    assert_eq!(parsed.amplitude, Some(Vec::new()));
    assert_eq!(parsed.key_hz, 440.0);
}

/// Reading robustness: the reserved area is not assumed zero — the corpus
/// carries `44100` as int32 and the ` speedwagon` marker there.
#[test]
fn nonzero_reserved_bytes_are_ignored() {
    let mut reserved = [0u8; 16];
    reserved[..4].copy_from_slice(&44_100i32.to_le_bytes());
    reserved[4..14].copy_from_slice(b" speedwago");
    reserved[14..16].copy_from_slice(b"n ");
    let bytes = frq_file(256, 345.868, &reserved, 1, &[(345.0, 7.0)]);

    let parsed = frq::parse(&bytes).unwrap();
    assert_eq!(parsed.key_hz, 345.868);
    assert_eq!(parsed.f0_hz, vec![345.0]);
    assert_eq!(parsed.amplitude, Some(vec![7.0]));
}

/// Reading robustness 4: `hop` is read, never assumed to be 256.
#[test]
fn hop_is_read_not_assumed() {
    let bytes = frq_file(512, 300.0, &[0; 16], 2, &[(300.0, 1.0), (0.0, 0.0)]);
    let parsed = frq::parse(&bytes).unwrap();
    assert_eq!(parsed.hop_samples, 512);

    let written = frq::to_bytes(&parsed);
    let (hop, ..) = decode(&written);
    assert_eq!(hop, 512, "the hop round-trips");
}

/// A hop the neutral table cannot express is a typed error.
#[test]
fn non_positive_hop_is_a_typed_error() {
    for hop in [0i32, -256] {
        let bytes = frq_file(hop, 440.0, &[0; 16], 0, &[]);
        assert_eq!(frq::parse(&bytes), Err(FrqError::InvalidHop { hop }));
    }
}

/// Writing: the exact 40-byte header (`FREQ0003`, hop, key, zeroed reserved,
/// count) followed by interleaved `(f0, amp)` f64 pairs.
#[test]
fn writer_layout_matches_the_spec() {
    let table = table(
        vec![100.0, 0.0, 220.5],
        Some(vec![1000.0, 0.0, 42.25]),
        440.0,
    );
    let bytes = frq::to_bytes(&table);
    assert_eq!(bytes.len(), HEADER_LEN + 3 * FRAME_LEN);

    let (hop, key, reserved, n, frames) = decode(&bytes);
    assert_eq!(hop, 256);
    assert_eq!(key, 440.0);
    assert_eq!(reserved, [0u8; 16], "reserved bytes are zeroed");
    assert_eq!(n, 3, "header N is the frame count");
    assert_eq!(
        frames,
        vec![(100.0, 1000.0), (0.0, 0.0), (220.5, 42.25)],
        "pairs are interleaved f0, amp, f0, amp, ..."
    );
}

/// A table without amplitude data (a format conversion) writes zero frames;
/// the format always has the field.
#[test]
fn amplitude_none_writes_zero_amplitudes() {
    let table = table(vec![200.0, 0.0], None, 200.0);
    let (.., frames) = decode(&frq::to_bytes(&table));
    assert_eq!(frames, vec![(200.0, 0.0), (0.0, 0.0)]);
}

/// The #8 all-unvoiced shape: f0 all `0.0`, key `0.0`, amplitude `0.0` — both
/// through bytes and through a real file.
#[test]
fn all_unvoiced_tables_keep_zero_key_and_amplitude() {
    let scratch = Scratch::new("unvoiced");
    let table = table(vec![0.0; 4], Some(vec![0.0; 4]), 0.0);

    let bytes = frq::to_bytes(&table);
    let (_, key, _, _, frames) = decode(&bytes);
    assert_eq!(key, 0.0);
    assert!(frames.iter().all(|&(f0, amp)| f0 == 0.0 && amp == 0.0));

    let path = scratch.join("a_wav.frq");
    frq::write(&path, &table).unwrap();
    let read_back = frq::read(&path).unwrap();
    assert_eq!(read_back, table);
    assert_eq!(read_back.key_hz, 0.0);
}

/// A full round-trip: bytes -> table -> bytes is stable, and the file helpers
/// agree with the byte helpers.
#[test]
fn round_trip_through_bytes_and_files() {
    let scratch = Scratch::new("roundtrip");
    let table = table(
        vec![261.63, 0.0, 440.0, 0.0, 220.0],
        Some(vec![32767.0, 0.0, 123.5, 0.25, 7082.8]),
        415.25,
    );

    let bytes = frq::to_bytes(&table);
    assert_eq!(frq::parse(&bytes).unwrap(), table);
    assert_eq!(frq::to_bytes(&frq::parse(&bytes).unwrap()), bytes);

    let path = scratch.join("nested/a_wav.frq");
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    frq::write(&path, &table).unwrap();
    assert_eq!(fs::read(&path).unwrap(), bytes);
    assert_eq!(frq::read(&path).unwrap(), table);
}

/// A missing file is an IO error, not a parse error.
#[test]
fn reading_a_missing_file_is_an_io_error() {
    let scratch = Scratch::new("missing");
    match frq::read(&scratch.join("no_wav.frq")) {
        Err(ReadError::Io(error)) => assert_eq!(error.kind(), io::ErrorKind::NotFound),
        other => panic!("expected NotFound, got {other:?}"),
    }
}
