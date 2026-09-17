//! Structural contract for `kira_frq_core::pmk` — the seven points of #9's
//! resolution, exercised on synthetic contours, plus the walk's frame lookup.

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use kira_frq_core::FrequencyTable;
use kira_frq_core::pmk;

/// A decoded pmk file, independent of the module's own layout knowledge.
struct Decoded {
    version: i16,
    avg: f64,
    entries: Vec<(i32, i32)>,
}

fn decode(bytes: &[u8]) -> Decoded {
    assert!(bytes.len() >= 14, "header is 14 bytes");
    let version = i16::from_le_bytes(bytes[0..2].try_into().unwrap());
    let avg = f64::from_le_bytes(bytes[2..10].try_into().unwrap());
    let count = i32::from_le_bytes(bytes[10..14].try_into().unwrap()) as usize;
    assert_eq!(bytes.len(), 14 + 8 * count, "size == 14 + 8*N");
    let entries = bytes[14..]
        .as_chunks::<8>()
        .0
        .iter()
        .map(|chunk| {
            (
                i32::from_le_bytes(chunk[0..4].try_into().unwrap()),
                i32::from_le_bytes(chunk[4..8].try_into().unwrap()),
            )
        })
        .collect();
    Decoded {
        version,
        avg,
        entries,
    }
}

fn table(hop_samples: u32, f0_hz: Vec<f64>) -> FrequencyTable {
    FrequencyTable {
        sample_rate: 44100,
        hop_samples,
        f0_hz,
        amplitude: None,
        key_hz: 0.0,
    }
}

/// A canonical-generation table for a wav of `length` samples: `N =
/// floor(L/256) + 1` frames, f0 from `f`.
fn contour(length: usize, f: impl Fn(usize) -> f64) -> FrequencyTable {
    let frames = length / 256 + 1;
    table(256, (0..frames).map(f).collect())
}

/// The #9 walk, transcribed from the resolution, as the oracle the writer must
/// reproduce entry for entry.
fn spec_walk(table: &FrequencyTable, length: usize) -> Vec<(i32, i32)> {
    assert!(!table.f0_hz.is_empty());
    let hop = f64::from(table.hop_samples);
    let mut pos = 0.0f64;
    let mut entries = Vec::new();
    loop {
        let index = ((pos / hop).floor() as usize).min(table.f0_hz.len() - 1);
        let f = table.f0_hz[index];
        let mut code = if !f.is_finite() || f <= 0.0 {
            pmk::UNVOICED_CODE
        } else {
            (44_100.0 / f).round() as i32
        };
        if !(pmk::MIN_VOICED_CODE..=pmk::MAX_VOICED_CODE).contains(&code) {
            code = pmk::UNVOICED_CODE;
        }
        let period = if code == pmk::UNVOICED_CODE {
            49.0
        } else {
            44_100.0 / f
        };
        // The emitted mark is rounded, so the stop is too: a next mark that
        // rounds to `L` is not below it.
        let next = pos + period;
        if next.round() >= length as f64 {
            return entries;
        }
        pos = next;
        entries.push((pos.round() as i32, code));
    }
}

/// Contours that between them exercise voicing, holes, out-of-range f0 and
/// frame transitions.
fn sweep_contours(length: usize) -> Vec<FrequencyTable> {
    vec![
        contour(length, |_| 0.0),
        contour(length, |_| 200.0),
        contour(length, |i| if i % 2 == 0 { 200.0 } else { 0.0 }),
        contour(length, |i| 100.0 + (i % 40) as f64 * 10.0),
        contour(length, |i| if i == 2 { f64::NAN } else { 150.0 }),
        contour(length, |i| if i == 3 { -5.0 } else { 220.0 }),
        contour(length, |i| if i == 4 { f64::INFINITY } else { 180.0 }),
    ]
}

/// Contract 1: header fields and file size.
#[test]
fn header_and_size_follow_the_format_contract() {
    for length in [0usize, 49, 50, 100, 300, 44_100 + 123, 87_040] {
        for table in sweep_contours(length) {
            let bytes = pmk::to_bytes(&table, length);
            let decoded = decode(&bytes);
            assert_eq!(decoded.version, 19, "version");
            assert!(
                (0.0..=513.0).contains(&decoded.avg) && decoded.avg.is_finite(),
                "avg {0} is not a plausible code mean",
                decoded.avg
            );
        }
    }
    assert_eq!(pmk::VERSION, 19);
}

/// Contract 2: the walk emits nothing only while no whole mark fits below `L`:
/// always at `L <= 49` (every step is at least 49), and for a voiced start
/// whose period reaches `L`. An unvoiced start always yields entries from
/// `L >= 50`.
#[test]
fn entry_count_is_zero_only_while_no_mark_fits_below_l() {
    for length in [0usize, 1, 48, 49] {
        for table in sweep_contours(length) {
            assert_eq!(
                decode(&pmk::to_bytes(&table, length)).entries.len(),
                0,
                "L = {length} must be header-only"
            );
        }
    }
    for length in [50usize, 98, 99, 100, 255, 256, 44_100] {
        let silent = decode(&pmk::to_bytes(&contour(length, |_| 0.0), length));
        assert!(!silent.entries.is_empty(), "L = {length} unvoiced start");
    }

    // A 100 Hz voiced start has a 441-sample first period: nothing fits in 50.
    let low = decode(&pmk::to_bytes(&contour(50, |_| 100.0), 50));
    assert!(low.entries.is_empty());
    // 200 Hz walks in 220.5-sample steps: the first mark is 220.5, which
    // rounds up to 221 == L and is therefore not emitted.
    assert!(
        decode(&pmk::to_bytes(&contour(221, |_| 200.0), 221))
            .entries
            .is_empty(),
        "the first rounded mark lands exactly on L"
    );
    assert_eq!(
        decode(&pmk::to_bytes(&contour(222, |_| 200.0), 222)).entries,
        vec![(221, 221)],
        "one sample longer and the mark fits"
    );
}

/// Contract 3: `pos_end` strictly increases and stays inside `(0, L)`.
#[test]
fn positions_strictly_increase_and_stay_below_l() {
    for length in [50usize, 300, 44_100 + 123, 87_040] {
        for table in sweep_contours(length) {
            let entries = decode(&pmk::to_bytes(&table, length)).entries;
            let mut previous = 0i32;
            for &(pos_end, _) in &entries {
                assert!(pos_end > previous, "pos_end {pos_end} after {previous}");
                assert!(
                    (pos_end as usize) < length,
                    "pos_end {pos_end} must stay below L = {length}"
                );
                previous = pos_end;
            }
        }
    }
}

/// Contract 4: every code is `49` or inside TIPS's detected range.
#[test]
fn codes_are_unvoiced_or_inside_the_tips_range() {
    for length in [50usize, 300, 44_100] {
        for table in sweep_contours(length) {
            for &(_, code) in &decode(&pmk::to_bytes(&table, length)).entries {
                assert!(
                    code == pmk::UNVOICED_CODE
                        || (pmk::MIN_VOICED_CODE..=pmk::MAX_VOICED_CODE).contains(&code),
                    "code {code} outside 49 or 50..=513"
                );
            }
        }
    }
}

/// f0 outside TIPS's 50..=513 period range becomes unvoiced with a 49-sample
/// walk: E2 (82.41 Hz, code 535) and anything above the range.
#[test]
fn out_of_range_f0_walks_as_unvoiced() {
    for f0 in [82.41, 70.0, 900.0] {
        let length = 5000;
        let entries = pmk::to_bytes(&contour(length, |_| f0), length);
        let decoded = decode(&entries);
        assert!(!decoded.entries.is_empty());
        assert_eq!(decoded.avg, 0.0, "{f0} Hz is entirely unvoiced");
        for (index, &(pos_end, code)) in decoded.entries.iter().enumerate() {
            assert_eq!(code, pmk::UNVOICED_CODE, "{f0} Hz");
            assert_eq!(pos_end, 49 * (index as i32 + 1), "{f0} Hz walk");
        }
    }

    for f0 in [882.0, 44100.0 / 513.0] {
        let length = 5000;
        let decoded = decode(&pmk::to_bytes(&contour(length, |_| f0), length));
        assert!(
            decoded.entries.iter().any(|&(_, code)| code != 49),
            "{f0} Hz must be voiced"
        );
    }
}

/// Contract 5: `pos_end[0] == code[0]`, for an unvoiced and a voiced start.
#[test]
fn first_position_equals_first_code() {
    let unvoiced = decode(&pmk::to_bytes(&contour(5000, |_| 0.0), 5000));
    assert_eq!(unvoiced.entries[0], (49, 49));

    let voiced = decode(&pmk::to_bytes(&contour(5000, |_| 200.0), 5000));
    assert_eq!(voiced.entries[0].0, voiced.entries[0].1);
    assert_eq!(voiced.entries[0], (221, 221));
}

/// Contract 6: `avg` is the exact f64 mean of the voiced codes, `0.0` when
/// nothing is voiced.
#[test]
fn avg_is_the_exact_voiced_code_mean() {
    for length in [50usize, 300, 44_100 + 123] {
        for table in sweep_contours(length) {
            let decoded = decode(&pmk::to_bytes(&table, length));
            let voiced: Vec<f64> = decoded
                .entries
                .iter()
                .filter(|&&(_, code)| code != pmk::UNVOICED_CODE)
                .map(|&(_, code)| code as f64)
                .collect();
            let expected = if voiced.is_empty() {
                0.0
            } else {
                voiced.iter().sum::<f64>() / voiced.len() as f64
            };
            assert_eq!(
                decoded.avg.to_bits(),
                expected.to_bits(),
                "avg must be the exact entry-weighted voiced mean"
            );
        }
    }

    let silent = decode(&pmk::to_bytes(&contour(44_100, |_| 0.0), 44_100));
    assert_eq!(silent.avg, 0.0);
    assert_eq!(
        decode(&pmk::to_bytes(&contour(49, |_| 200.0), 49)).avg,
        0.0,
        "header-only file has no voiced entries"
    );
}

/// Contract 7: the writer reproduces the #9 walk exactly (deterministic f64),
/// and the walk stops before `L` with no next step fitting.
#[test]
fn walk_reproduces_the_spec_and_is_deterministic() {
    for length in [50usize, 300, 44_100 + 123, 87_040] {
        for table in sweep_contours(length) {
            let bytes = pmk::to_bytes(&table, length);
            assert_eq!(
                decode(&bytes).entries,
                spec_walk(&table, length),
                "L = {length}"
            );
            assert_eq!(bytes, pmk::to_bytes(&table, length), "deterministic f64");
        }
    }
}

/// A constant 300 Hz tone has an exactly representable 147-sample period, so
/// the walk is exactly `147*i` and stops at the last mark below `L`.
#[test]
fn constant_tone_walks_one_period_at_a_time() {
    let length = 44_100;
    let decoded = decode(&pmk::to_bytes(&contour(length, |_| 300.0), length));
    assert_eq!(decoded.entries.len(), 299, "147*300 == L is not emitted");
    for (index, &(pos_end, code)) in decoded.entries.iter().enumerate() {
        assert_eq!(pos_end, 147 * (index as i32 + 1));
        assert_eq!(code, 147);
    }
    assert_eq!(decoded.avg, 147.0);
}

/// A constant 200 Hz tone has period 220.5; the emitted positions alternate
/// their delta to carry the fraction (error diffusion).
#[test]
fn fractional_period_error_diffuses_without_drift() {
    let length = 44_100;
    let decoded = decode(&pmk::to_bytes(&contour(length, |_| 200.0), length));
    assert_eq!(decoded.entries.len(), 199, "220.5*200 == L is not emitted");
    let mut previous = 0;
    for (index, &(pos_end, _)) in decoded.entries.iter().enumerate() {
        let ideal = 220.5 * (index as f64 + 1.0);
        assert!(
            (pos_end as f64 - ideal).abs() <= 0.5,
            "position {pos_end} drifts from {ideal}"
        );
        let delta = pos_end - previous;
        assert!(delta == 220 || delta == 221, "delta {delta}");
        previous = pos_end;
    }
    assert_eq!(decoded.entries.last().unwrap().0, 43_880);
}

/// The frame lookup is floor on the table's own hop, and the last frame is the
/// belt for positions past the table.
#[test]
fn frame_lookup_is_floor_on_the_tables_hop() {
    // hop 128: the first 220.5-sample step lands in frame 1, not frame 0.
    let narrow = table(128, vec![200.0, 441.0]);
    let entries = decode(&pmk::to_bytes(&narrow, 1000)).entries;
    assert_eq!(
        entries,
        vec![
            (221, 221),
            (321, 100),
            (421, 100),
            (521, 100),
            (621, 100),
            (721, 100),
            (821, 100),
            (921, 100),
        ]
    );

    // hop 256 over the same f0: the 220.5-sample step stays inside frame 0
    // until the second mark crosses the 256-sample boundary.
    let wide = table(256, vec![200.0, 441.0]);
    let first = decode(&pmk::to_bytes(&wide, 1000)).entries;
    assert_eq!(&first[..2], &[(221, 221), (441, 221)]);
    assert_eq!(
        &first[2..],
        &[(541, 100), (641, 100), (741, 100), (841, 100), (941, 100)]
    );

    // A frame boundary exactly on a step: 172.265625 Hz == 44100/256.
    let boundary = table(256, vec![172.265625, 344.53125]);
    let entries = decode(&pmk::to_bytes(&boundary, 1000)).entries;
    assert_eq!(
        entries,
        vec![
            (256, 256),
            (384, 128),
            (512, 128),
            (640, 128),
            (768, 128),
            (896, 128)
        ]
    );

    // A table shorter than the walk needs keeps reading its last frame.
    let short = table(256, vec![200.0]);
    let entries = decode(&pmk::to_bytes(&short, 1000)).entries;
    assert_eq!(entries.len(), 4, "220.5*4 < 1000 <= 220.5*5");
    assert!(entries.iter().all(|&(_, code)| code == 221));
}

/// An empty table cannot voice anything, so the walk is exact 49s.
#[test]
fn empty_table_walks_as_unvoiced() {
    let decoded = decode(&pmk::to_bytes(&table(256, vec![]), 1000));
    assert_eq!(decoded.entries.len(), 1000 / 49);
    assert_eq!(decoded.entries[0], (49, 49));
    assert_eq!(decoded.avg, 0.0);
}

/// The whole-file walk of silence matches the black-box TIPS probe: 899 marks,
/// the last at 44,051 (`900*49 == L` is not emitted).
#[test]
fn silence_walk_matches_the_engine_probe() {
    let decoded = decode(&pmk::to_bytes(&contour(44_100, |_| 0.0), 44_100));
    assert_eq!(decoded.entries.len(), 899);
    assert_eq!(decoded.entries.last().unwrap(), &(44_051, 49));
}

/// `write` lands the same bytes on disk as `to_bytes`.
#[test]
fn write_lands_the_encoded_bytes() {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let path: PathBuf = std::env::temp_dir().join(format!(
        "kira-frq-pmk-{}-{}.pmk",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let table = contour(5000, |_| 200.0);
    let expected = pmk::to_bytes(&table, 5000);
    pmk::write(&path, &table, 5000).unwrap();
    assert_eq!(fs::read(&path).unwrap(), expected);
    let _ = fs::remove_file(path);
}
