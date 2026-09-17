//! Structural contract for `kira_frq_core::mrq` — the seven points of #10's
//! resolution, exercised on byte fixtures and real files in a scratch folder.

use std::ffi::OsStr;
use std::fs;
use std::path::Path;
use std::time::{Duration, UNIX_EPOCH};

use kira_frq_core::FrequencyTable;
use kira_frq_core::mrq::{self, Desc, DescError, Entry, MergeState, Sharing};

mod common;
use common::Scratch;

const TIMESTAMP: i32 = 1_700_000_000;

fn key(name: &str) -> Vec<u16> {
    name.encode_utf16().collect()
}

/// A canonical generation table for a wav of `length` samples: frames cover
/// `[i*256, (i+1)*256)` and the trailing partial frame is forced unvoiced (#8).
fn table_for(length: usize) -> FrequencyTable {
    let hop = 256usize;
    let frames = length / hop + 1;
    let f0_hz = (0..frames)
        .map(|i| {
            if i % 4 == 2 || i + 1 == frames {
                0.0
            } else {
                110.0 + i as f64
            }
        })
        .collect();
    FrequencyTable {
        sample_rate: 44100,
        hop_samples: hop as u32,
        f0_hz,
        amplitude: None,
        key_hz: 0.0,
    }
}

/// One raw entry: `nfilename`, UTF-16 name, `size`, trunk.
fn raw_entry(name_units: &[u16], size: usize, fill: u8) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&(name_units.len() as i32).to_le_bytes());
    for unit in name_units {
        out.extend_from_slice(&unit.to_le_bytes());
    }
    out.extend_from_slice(&(size as i32).to_le_bytes());
    out.extend(std::iter::repeat_n(fill, size));
    out
}

fn foreign_entry(name: &str, size: usize, fill: u8) -> Vec<u8> {
    raw_entry(&key(name), size, fill)
}

/// A whole v2 file from raw entries.
fn mrq_file(entries: &[Vec<u8>]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"mrq ");
    out.extend_from_slice(&2i32.to_le_bytes());
    out.extend_from_slice(&(entries.len() as i32).to_le_bytes());
    for entry in entries {
        out.extend_from_slice(entry);
    }
    out
}

/// An [`Entry`] serialized back to its raw form.
fn encode_entry(entry: &Entry) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&(entry.name().len() as i32).to_le_bytes());
    for unit in entry.name() {
        out.extend_from_slice(&unit.to_le_bytes());
    }
    out.extend_from_slice(&(entry.raw().len() as i32).to_le_bytes());
    out.extend_from_slice(entry.raw());
    out
}

/// Independent raw decode of a whole file.
fn parse_raw(bytes: &[u8]) -> Vec<(Vec<u16>, Vec<u8>)> {
    assert_eq!(&bytes[..4], b"mrq ");
    assert_eq!(i32::from_le_bytes(bytes[4..8].try_into().unwrap()), 2);
    let count = i32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    let mut pos = 12usize;
    let mut out = Vec::new();
    for _ in 0..count {
        let name_len = i32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        let name = bytes[pos..pos + 2 * name_len]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|chunk| u16::from_le_bytes(*chunk))
            .collect();
        pos += 2 * name_len;
        let size = i32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        out.push((name, bytes[pos..pos + size].to_vec()));
        pos += size;
    }
    assert_eq!(pos, bytes.len(), "the file must be consumed exactly");
    out
}

/// Contract 1: parse round-trips byte-identically and trusts the stored `size`
/// (corpus entries can carry sizes larger than `20 + 4*nf0`).
#[test]
fn parse_trusts_stored_size_and_round_trips() {
    let weird = foreign_entry("foreign.wav", 308, 0xA5);
    let file = mrq_file(std::slice::from_ref(&weird));

    let desc = Desc::parse(&file).unwrap();
    assert_eq!(desc.len(), 1);
    assert_eq!(desc.entries()[0].name(), key("foreign.wav"));
    assert_eq!(desc.entries()[0].raw().len(), 308);
    assert_eq!(desc.to_bytes(), file);

    let trailing = [file.clone(), vec![0xFF]].concat();
    assert_eq!(
        Desc::parse(&trailing),
        Err(DescError::TrailingData { extra: 1 })
    );
    assert_eq!(Desc::parse(b"nope"), Err(DescError::BadMagic));
    assert_eq!(
        Desc::parse(&[b"mrq ".to_vec(), 2i32.to_le_bytes().to_vec()].concat()),
        Err(DescError::Truncated),
        "missing entry count"
    );

    let mut version_one = file.clone();
    version_one[4..8].copy_from_slice(&1i32.to_le_bytes());
    assert_eq!(
        Desc::parse(&version_one),
        Err(DescError::UnsupportedVersion { version: 1 })
    );

    let truncated = &file[..file.len() - 1];
    assert_eq!(
        Desc::parse(truncated),
        Err(DescError::SizeOverrun {
            entry: 0,
            size: 308
        })
    );
}

/// Contract 2: written entries follow the v2 layout exactly.
#[test]
fn written_entries_match_the_format_contract() {
    for length in [256usize, 44_100 + 123, 87_040] {
        let table = table_for(length);
        let nf0 = length / 256;
        let entry = Entry::from_table(&key("voice.wav"), &table, TIMESTAMP).unwrap();
        let mut desc = Desc::empty();
        assert!(!desc.upsert(entry));
        let bytes = desc.to_bytes();

        let raw = parse_raw(&bytes);
        assert_eq!(raw.len(), 1);
        let (name, trunk) = &raw[0];
        assert_eq!(name, &key("voice.wav"));
        assert_eq!(trunk.len(), 20 + 4 * nf0, "size == 20 + 4*nf0");

        assert_eq!(
            i32::from_le_bytes(trunk[0..4].try_into().unwrap()),
            nf0 as i32
        );
        assert_eq!(i32::from_le_bytes(trunk[4..8].try_into().unwrap()), 44100);
        assert_eq!(i32::from_le_bytes(trunk[8..12].try_into().unwrap()), 256);
        for i in 0..nf0 {
            let stored = f32::from_le_bytes(trunk[12 + 4 * i..16 + 4 * i].try_into().unwrap());
            let source = table.f0_hz[i];
            let expected = if source > 0.0 { source as f32 } else { 0.0 };
            assert_eq!(stored.to_bits(), expected.to_bits(), "f0 frame {i}");
        }
        let tail = 12 + 4 * nf0;
        assert_eq!(
            i32::from_le_bytes(trunk[tail..tail + 4].try_into().unwrap()),
            TIMESTAMP
        );
        assert_eq!(
            i32::from_le_bytes(trunk[tail + 4..tail + 8].try_into().unwrap()),
            0,
            "modified == 0"
        );
    }
}

/// Contract 3: merge keeps foreign entries and tombstones byte-identical,
/// preserves order, appends new entries and never decrements the count.
#[test]
fn merge_preserves_foreign_entries_tombstones_and_order() {
    let foreign_a = foreign_entry("A.wav", 308, 0x11);
    let b_old = Entry::from_table(&key("B.wav"), &table_for(44_100), 100).unwrap();
    let tombstone = raw_entry(&[], 28, 0xEE);
    let foreign_d = foreign_entry("D.wav", 20, 0x22);
    let file = mrq_file(&[
        foreign_a.clone(),
        encode_entry(&b_old),
        tombstone.clone(),
        foreign_d.clone(),
    ]);

    let mut desc = Desc::parse(&file).unwrap();
    assert_eq!(desc.len(), 4);
    assert!(desc.has(&key("B.wav")));
    assert!(!desc.has(&key("missing.wav")));
    assert!(!desc.has(&[]), "tombstones never match");

    let b_new = Entry::from_table(&key("B.wav"), &table_for(20_000), 200).unwrap();
    let c_new = Entry::from_table(&key("C.wav"), &table_for(30_000), 300).unwrap();
    assert!(desc.upsert(b_new), "B.wav is replaced in place");
    assert!(!desc.upsert(c_new), "C.wav is appended");
    assert_eq!(desc.len(), 5);

    let merged = desc.to_bytes();
    assert_eq!(
        i32::from_le_bytes(merged[8..12].try_into().unwrap()),
        5,
        "entry_count never decrements"
    );
    let raw = parse_raw(&merged);
    assert_eq!(raw[0].0, key("A.wav"));
    assert_eq!(
        raw[0].1,
        foreign_a[4 + 2 * key("A.wav").len() + 4..].to_vec()
    );
    assert_eq!(raw[1].0, key("B.wav"));
    assert_ne!(raw[1].1, b_old.raw(), "B.wav got the new f0");
    assert!(raw[2].0.is_empty());
    assert_eq!(raw[2].1, tombstone[4 + 4..].to_vec());
    assert_eq!(raw[3].0, key("D.wav"));
    assert_eq!(
        raw[3].1,
        foreign_d[4 + 2 * key("D.wav").len() + 4..].to_vec()
    );
    assert_eq!(raw[4].0, key("C.wav"));
}

/// Contract 4: with a frozen clock a repeated upsert rewrites nothing.
#[test]
fn frozen_clock_rewrites_are_byte_identical() {
    let first = Entry::from_table(&key("B.wav"), &table_for(44_100), 500).unwrap();
    let second = first.clone();

    let mut once = Desc::empty();
    once.upsert(first);
    let mut twice = Desc::parse(&once.to_bytes()).unwrap();
    twice.upsert(second);
    assert_eq!(once.to_bytes(), twice.to_bytes());

    let reopened = Desc::parse(&once.to_bytes()).unwrap();
    assert_eq!(reopened, once);
}

/// Contract 5: corrupt input is renamed aside, reported, and replaced by a
/// fresh file that parses.
#[test]
fn corrupt_files_are_backed_up_and_replaced() {
    let scratch = Scratch::new("corrupt");
    let cases: [(&str, Vec<u8>); 4] = [
        ("bad magic", b"nope, not an mrq file".to_vec()),
        (
            "unsupported version",
            [
                b"mrq ".to_vec(),
                1i32.to_le_bytes().to_vec(),
                0i32.to_le_bytes().to_vec(),
            ]
            .concat(),
        ),
        (
            "truncated entry",
            [
                b"mrq ".to_vec(),
                2i32.to_le_bytes().to_vec(),
                1i32.to_le_bytes().to_vec(),
            ]
            .concat(),
        ),
        (
            "size overrun",
            mrq_file(&[{
                let mut entry = raw_entry(&key("x.wav"), 999, 0x00);
                entry.truncate(entry.len() - 500);
                entry
            }]),
        ),
    ];

    for (tag, corrupt) in cases {
        let path = scratch.join(&format!("{tag}/desc.mrq"));
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, &corrupt).unwrap();

        let outcome = mrq::open_for_merge(&path).unwrap();
        let MergeState::Recovered { backup, error } = &outcome.state else {
            panic!("{tag}: expected recovery, got {:?}", outcome.state);
        };
        assert!(outcome.desc.is_empty());
        assert!(!path.exists(), "{tag}: the corrupt file is gone");
        assert_eq!(fs::read(backup).unwrap(), corrupt, "{tag}: backup content");
        let backup_name = backup.file_name().unwrap().to_string_lossy().into_owned();
        assert!(
            backup_name.starts_with("desc.mrq.corrupt-"),
            "{tag}: backup name {backup_name:?}"
        );
        assert!(!error.to_string().is_empty());

        outcome.desc.write(&path).unwrap();
        assert_eq!(Desc::read(&path).unwrap(), Some(Desc::empty()));
    }

    let missing = scratch.join("missing/desc.mrq");
    fs::create_dir_all(missing.parent().unwrap()).unwrap();
    let outcome = mrq::open_for_merge(&missing).unwrap();
    assert_eq!(outcome.state, MergeState::Missing);
    assert!(
        !missing.exists(),
        "opening for merge does not create the file"
    );

    let good = scratch.join("good/desc.mrq");
    fs::create_dir_all(good.parent().unwrap()).unwrap();
    let fixture = mrq_file(&[foreign_entry("A.wav", 20, 0x33)]);
    fs::write(&good, &fixture).unwrap();
    let outcome = mrq::open_for_merge(&good).unwrap();
    assert_eq!(outcome.state, MergeState::Loaded);
    assert_eq!(outcome.desc, Desc::parse(&fixture).unwrap());
}

/// Contract 6: sub-hop wavs produce no entry; the first complete window does.
#[test]
fn sub_hop_wavs_have_no_entry() {
    for length in [0usize, 1, 255] {
        assert!(
            Entry::from_table(&key("tiny.wav"), &table_for(length), TIMESTAMP).is_none(),
            "L = {length}"
        );
    }
    let entry = Entry::from_table(&key("tiny.wav"), &table_for(256), TIMESTAMP).unwrap();
    let trunk = entry.raw();
    assert_eq!(i32::from_le_bytes(trunk[0..4].try_into().unwrap()), 1);
}

/// Contract 7: the sharing flag adds the CP932 name on a non-932 code page and
/// stays a no-op on 932; both names are required and both are updated.
#[test]
fn sharing_flag_writes_both_names_and_one_sided_counts_as_absent() {
    // `local` is what CP932 bytes of the intended name look like decoded as
    // CP936 — the mojibake state a bank written under a Chinese locale has.
    const JAPANESE: &str = "_\u{3042}\u{304b}\u{3055}\u{305f}\u{306a}.wav";
    const LOCAL: &str = "_\u{5041}\u{5050}\u{505d}\u{5068}\u{5074}.wav";

    let sharing = Sharing::new(936);
    assert_eq!(sharing.code_page(), 936);
    assert_eq!(sharing.japanese_key(&key(LOCAL)), Some(key(JAPANESE)));
    assert_eq!(
        mrq::entry_keys(OsStr::new(LOCAL), Some(&sharing)),
        vec![key(LOCAL), key(JAPANESE)],
        "local first, then the Japanese-side key"
    );
    assert_eq!(
        mrq::entry_keys(OsStr::new("voice.wav"), Some(&sharing)),
        vec![key("voice.wav")],
        "ASCII names convert to themselves, so there is nothing extra to write"
    );

    let native = Sharing::new(932);
    assert_eq!(native.japanese_key(&key(LOCAL)), None);
    assert_eq!(
        mrq::entry_keys(OsStr::new(LOCAL), Some(&native)),
        vec![key(LOCAL)]
    );
    assert_eq!(
        mrq::entry_keys(OsStr::new(LOCAL), None),
        vec![key(LOCAL)],
        "flag off: the local name alone decides"
    );

    let keys = mrq::entry_keys(OsStr::new(LOCAL), Some(&sharing));
    let table = table_for(44_100);

    let local_only = {
        let mut desc = Desc::empty();
        desc.upsert(Entry::from_table(&key(LOCAL), &table, 100).unwrap());
        desc
    };
    assert!(
        !keys.iter().all(|k| local_only.has(k)),
        "a one-sided local entry counts as absent with the flag on"
    );

    let japanese_only = {
        let mut desc = Desc::empty();
        desc.upsert(Entry::from_table(&key(JAPANESE), &table, 100).unwrap());
        desc
    };
    assert!(!keys.iter().all(|k| japanese_only.has(k)));

    let mut both = local_only.clone();
    for k in &keys {
        both.upsert(Entry::from_table(k, &table, 100).unwrap());
    }
    assert!(keys.iter().all(|k| both.has(k)));
    assert_eq!(both.len(), 2);
    let raw = parse_raw(&both.to_bytes());
    assert_eq!(raw[0].1, raw[1].1, "same f0 in both entries");

    let regenerated = {
        let mut desc = both.clone();
        for k in &keys {
            desc.upsert(Entry::from_table(k, &table_for(20_000), 200).unwrap());
        }
        desc
    };
    assert_eq!(regenerated.len(), 2, "no duplicates on regeneration");
    let raw = parse_raw(&regenerated.to_bytes());
    assert_eq!(raw[0].0, key(LOCAL), "replaced in place");
    assert_eq!(raw[1].0, key(JAPANESE));
    assert_ne!(raw[0].1, both.entries()[0].raw());

    let mut other_side = japanese_only;
    for k in &keys {
        other_side.upsert(Entry::from_table(k, &table, 100).unwrap());
    }
    let raw = parse_raw(&other_side.to_bytes());
    assert_eq!(raw[0].0, key(JAPANESE));
    assert_eq!(raw[1].0, key(LOCAL), "the missing side is appended");
    assert_eq!(other_side.len(), 2);
}

/// `timestamp = max(now, floor(wav_mtime))`, with pre-epoch times clamped to 0.
#[test]
fn entry_timestamp_is_the_later_whole_second() {
    let epoch_plus = |secs: u64| UNIX_EPOCH + Duration::from_secs(secs);
    assert_eq!(
        mrq::entry_timestamp(epoch_plus(1000), epoch_plus(2000)),
        2000
    );
    assert_eq!(
        mrq::entry_timestamp(epoch_plus(2000), epoch_plus(1000)),
        2000
    );
    assert_eq!(mrq::entry_timestamp(epoch_plus(0), epoch_plus(0)), 0);

    let before_epoch = UNIX_EPOCH.checked_sub(Duration::from_secs(1)).unwrap();
    assert_eq!(mrq::entry_timestamp(before_epoch, epoch_plus(7)), 7);
}

/// The per-folder path is `folder/desc.mrq`.
#[test]
fn desc_path_is_the_folder_file() {
    assert_eq!(
        mrq::desc_path(Path::new("/bank/A2")),
        Path::new("/bank/A2/desc.mrq")
    );
}
