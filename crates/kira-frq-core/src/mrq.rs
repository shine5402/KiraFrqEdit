//! moresampler `desc.mrq`: per-folder f0 store.
//!
//! Format spec: `docs/research/mrq-format.md`; merge, corrupt-file and sharing
//! policy was settled in #10 and lands here (#16).
//!
//! A [`Desc`] keeps every entry's bytes verbatim, so a merge rewrites only the
//! entries for the wavs in the run: foreign entries, tombstones and entry order
//! survive untouched, new entries are appended and the entry count never
//! decreases (ADR 0001 keeps the native type for exactly this). Parsing trusts
//! each entry's stored `size` — some corpus entries carry sizes larger than
//! `20 + 4*nf0`.
//!
//! A corrupt or unsupported `desc.mrq` is renamed aside by [`open_for_merge`] —
//! never merged into, never silently clobbered.
//!
//! The opt-in sharing flag adds a second entry per wav keyed by the
//! Japanese-side name; [`Sharing`] carries the machine's 8-bit code page and is
//! injectable for tests.

use std::ffi::OsStr;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::FrequencyTable;

/// The one file name moresampler uses per voice folder.
pub const FILE_NAME: &str = "desc.mrq";

/// The `desc.mrq` path inside `folder`.
pub fn desc_path(folder: &Path) -> PathBuf {
    folder.join(FILE_NAME)
}

/// A folder's `desc.mrq` in memory: an ordered list of entries, each carrying
/// its name and its raw data bytes exactly as stored.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Desc {
    entries: Vec<Entry>,
}

/// One `desc.mrq` entry. The name is kept as UTF-16 code units (the on-disk
/// filename verbatim) and the trunk as raw bytes, so untouched entries are
/// rewritten byte-identically.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    name: Vec<u16>,
    trunk: Vec<u8>,
}

impl Entry {
    /// Build an entry for `name` from a generated table.
    ///
    /// The table must use the canonical generation grid (`N = floor(L/hop) + 1`,
    /// trailing partial frame included and forced unvoiced by #8); the entry
    /// keeps the complete windows only, so `nf0 = f0_hz.len() - 1` and sub-hop
    /// wavs (`nf0 == 0`) produce no entry (#8).
    ///
    /// `timestamp` follows #4/#10: `max(now, floor(wav_mtime))` — see
    /// [`entry_timestamp`]; `modified` is always `0` (the hand-edit marker).
    pub fn from_table(name: &[u16], table: &FrequencyTable, timestamp: i32) -> Option<Self> {
        let nf0 = table.f0_hz.len().checked_sub(1)?;
        if nf0 == 0 {
            return None;
        }
        let mut trunk = Vec::with_capacity(20 + 4 * nf0);
        trunk.extend_from_slice(&(nf0 as i32).to_le_bytes());
        trunk.extend_from_slice(&(table.sample_rate as i32).to_le_bytes());
        trunk.extend_from_slice(&(table.hop_samples as i32).to_le_bytes());
        for &f0 in &table.f0_hz[..nf0] {
            let stored = if f0.is_finite() && f0 > 0.0 {
                f0 as f32
            } else {
                0.0
            };
            trunk.extend_from_slice(&stored.to_le_bytes());
        }
        trunk.extend_from_slice(&timestamp.to_le_bytes());
        trunk.extend_from_slice(&0i32.to_le_bytes());
        Some(Self {
            name: name.to_vec(),
            trunk,
        })
    }

    /// The entry name: the bare wav filename as code units, verbatim.
    pub fn name(&self) -> &[u16] {
        &self.name
    }

    /// The raw data bytes after the stored `size` field.
    pub fn raw(&self) -> &[u8] {
        &self.trunk
    }
}

impl Desc {
    /// A fresh v2 `Desc` with no entries.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Parse a complete v2 file; every byte must belong to an entry.
    ///
    /// The stored `size` of each entry is trusted for skipping, so entries
    /// whose trunk is longer than `20 + 4*nf0` survive unchanged.
    pub fn parse(bytes: &[u8]) -> Result<Self, DescError> {
        if bytes.get(..4) != Some(b"mrq ") {
            return Err(DescError::BadMagic);
        }
        let version = read_i32(bytes, 4)?;
        if version != 2 {
            return Err(DescError::UnsupportedVersion { version });
        }
        let count = read_i32(bytes, 8)?;
        if count < 0 {
            return Err(DescError::Truncated);
        }
        let mut entries = Vec::with_capacity((count as usize).min(bytes.len() / 8));
        let mut pos = 12;
        for index in 0..count as usize {
            let name_len = read_i32(bytes, pos)?;
            pos += 4;
            if name_len < 0 {
                return Err(DescError::Truncated);
            }
            let name_len = name_len as usize;
            let name_bytes_len = name_len.checked_mul(2).ok_or(DescError::Truncated)?;
            let name_end = pos
                .checked_add(name_bytes_len)
                .ok_or(DescError::Truncated)?;
            let name_bytes = bytes.get(pos..name_end).ok_or(DescError::Truncated)?;
            let name = name_bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|chunk| u16::from_le_bytes(*chunk))
                .collect();
            pos = name_end;

            let size = read_i32(bytes, pos)?;
            pos += 4;
            if size < 0 {
                return Err(DescError::SizeOverrun { entry: index, size });
            }
            let trunk_end = pos
                .checked_add(size as usize)
                .ok_or(DescError::SizeOverrun { entry: index, size })?;
            let trunk = bytes
                .get(pos..trunk_end)
                .ok_or(DescError::SizeOverrun { entry: index, size })?;
            entries.push(Entry {
                name,
                trunk: trunk.to_vec(),
            });
            pos = trunk_end;
        }
        if pos != bytes.len() {
            return Err(DescError::TrailingData {
                extra: bytes.len() - pos,
            });
        }
        Ok(Self { entries })
    }

    /// Read and parse `path`; `Ok(None)` when the file does not exist.
    pub fn read(path: &Path) -> Result<Option<Self>, ReadError> {
        match fs::read(path) {
            Ok(bytes) => Desc::parse(&bytes).map(Some).map_err(ReadError::Parse),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(ReadError::Io(error)),
        }
    }

    /// Serialize the whole file, untouched entries byte-identically.
    pub fn to_bytes(&self) -> Vec<u8> {
        let capacity = 12
            + self
                .entries
                .iter()
                .map(|entry| 8 + 2 * entry.name.len() + entry.trunk.len())
                .sum::<usize>();
        let mut out = Vec::with_capacity(capacity);
        out.extend_from_slice(b"mrq ");
        out.extend_from_slice(&2i32.to_le_bytes());
        out.extend_from_slice(&(self.entries.len() as i32).to_le_bytes());
        for entry in &self.entries {
            out.extend_from_slice(&(entry.name.len() as i32).to_le_bytes());
            for unit in &entry.name {
                out.extend_from_slice(&unit.to_le_bytes());
            }
            out.extend_from_slice(&(entry.trunk.len() as i32).to_le_bytes());
            out.extend_from_slice(&entry.trunk);
        }
        out
    }

    /// Write the serialized file to `path`.
    pub fn write(&self, path: &Path) -> io::Result<()> {
        fs::write(path, self.to_bytes())
    }

    /// Whether an entry with exactly this name exists. Tombstones (empty
    /// names) never match.
    pub fn has(&self, name: &[u16]) -> bool {
        !name.is_empty() && self.entries.iter().any(|entry| entry.name == name)
    }

    /// Replace the entry with this name in place, or append it; returns `true`
    /// when an existing entry was replaced. Entry order is preserved and the
    /// count never decreases.
    pub fn upsert(&mut self, entry: Entry) -> bool {
        match self
            .entries
            .iter_mut()
            .find(|existing| existing.name == entry.name)
        {
            Some(slot) => {
                *slot = entry;
                true
            }
            None => {
                self.entries.push(entry);
                false
            }
        }
    }

    /// The entries as stored, in file order.
    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    /// The number of entries, tombstones included.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the file has no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Everything about a `desc.mrq` that makes it unusable for merging.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DescError {
    /// The magic is not `mrq `.
    BadMagic,
    /// File-level version is not 2 (v1 files are not merged either).
    UnsupportedVersion { version: i32 },
    /// A read ran past the end of the file (or a length field was negative).
    Truncated,
    /// An entry's stored `size` runs past the end of the file.
    SizeOverrun { entry: usize, size: i32 },
    /// Bytes remain after the last entry.
    TrailingData { extra: usize },
}

impl std::fmt::Display for DescError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DescError::BadMagic => write!(f, "not a desc.mrq file (bad magic)"),
            DescError::UnsupportedVersion { version } => write!(
                f,
                "unsupported desc.mrq version {version} (only v2 can be merged)"
            ),
            DescError::Truncated => write!(f, "truncated desc.mrq data"),
            DescError::SizeOverrun { entry, size } => write!(
                f,
                "desc.mrq entry {entry} with stored size {size} overruns the file"
            ),
            DescError::TrailingData { extra } => write!(
                f,
                "desc.mrq has {extra} trailing byte(s) after the last entry"
            ),
        }
    }
}

impl std::error::Error for DescError {}

/// [`Desc::read`] can fail either reading or parsing.
#[derive(Debug)]
pub enum ReadError {
    Io(io::Error),
    Parse(DescError),
}

impl std::fmt::Display for ReadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReadError::Io(error) => write!(f, "{error}"),
            ReadError::Parse(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for ReadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ReadError::Io(error) => Some(error),
            ReadError::Parse(error) => Some(error),
        }
    }
}

/// The state [`open_for_merge`] found the file in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MergeState {
    /// No file existed; the fresh [`Desc`] starts empty.
    Missing,
    /// An existing file parsed and is ready to merge into.
    Loaded,
    /// The file could not be parsed, so it was renamed aside to `backup` and
    /// the fresh [`Desc`] starts empty.
    Recovered { backup: PathBuf, error: DescError },
}

/// A [`Desc`] ready for merge-writes plus what opening it found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MergeOutcome {
    pub desc: Desc,
    pub state: MergeState,
}

/// Open a folder's `desc.mrq` for a merge-write.
///
/// A corrupt/unsupported file is renamed to `desc.mrq.corrupt-<UTC seconds>`
/// (with `-1`, `-2`, … appended if that is taken) and reported through
/// [`MergeState::Recovered`], so the caller can warn; the returned fresh [`Desc`]
/// is then written as a new v2 file. Never merge into a corrupt file.
pub fn open_for_merge(path: &Path) -> io::Result<MergeOutcome> {
    match Desc::read(path) {
        Ok(Some(desc)) => Ok(MergeOutcome {
            desc,
            state: MergeState::Loaded,
        }),
        Ok(None) => Ok(MergeOutcome {
            desc: Desc::empty(),
            state: MergeState::Missing,
        }),
        Err(ReadError::Io(error)) => Err(error),
        Err(ReadError::Parse(error)) => {
            let backup = unique_backup_path(path);
            fs::rename(path, &backup)?;
            Ok(MergeOutcome {
                desc: Desc::empty(),
                state: MergeState::Recovered { backup, error },
            })
        }
    }
}

/// The v2 `timestamp` for a just-processed wav (#4/#10): the later of `now` and
/// the wav's mtime in whole seconds. `timestamp == 0` is reserved for explicit
/// pin/hand-edit actions, so pre-epoch times clamp to 0.
pub fn entry_timestamp(now: SystemTime, wav_mtime: SystemTime) -> i32 {
    let now = whole_seconds(now);
    let wav_mtime = whole_seconds(wav_mtime);
    now.max(wav_mtime) as i32
}

fn whole_seconds(time: SystemTime) -> i64 {
    match time.duration_since(UNIX_EPOCH) {
        Ok(duration) => (duration.as_secs().min(i32::MAX as u64)) as i64,
        Err(_) => 0,
    }
}

fn unique_backup_path(path: &Path) -> PathBuf {
    let file_name = path
        .file_name()
        .map(OsStr::to_os_string)
        .unwrap_or_else(|| FILE_NAME.into());
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0);
    let mut candidate = path.to_path_buf();
    for attempt in 0u32.. {
        let mut name = file_name.clone();
        if attempt == 0 {
            name.push(format!(".corrupt-{stamp}"));
        } else {
            name.push(format!(".corrupt-{stamp}-{attempt}"));
        }
        candidate.set_file_name(name);
        if !candidate.exists() {
            return candidate;
        }
    }
    unreachable!("backup name search is unbounded")
}

fn read_i32(bytes: &[u8], offset: usize) -> Result<i32, DescError> {
    let end = offset.checked_add(4).ok_or(DescError::Truncated)?;
    let slice = bytes.get(offset..end).ok_or(DescError::Truncated)?;
    Ok(i32::from_le_bytes(slice.try_into().expect("four bytes")))
}

/// The machine's 8-bit code page, as the sharing flag needs it: encoding a
/// filename with it and decoding those bytes as CP932 yields the key a
/// Japanese-region moresampler computes for the same wav (#10).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sharing {
    code_page: u32,
}

impl Sharing {
    /// An explicit code page — tests inject one, and the CLI/GUI flag can
    /// override the machine default.
    pub fn new(code_page: u32) -> Self {
        Self { code_page }
    }

    /// The machine's ANSI code page (`GetACP` on Windows); other platforms have
    /// no 8-bit layer, so the flag is a no-op there (code page 932).
    pub fn native() -> Self {
        #[cfg(windows)]
        {
            // SAFETY: `GetACP` takes no arguments and cannot fail.
            Self::new(unsafe { GetACP() })
        }
        #[cfg(not(windows))]
        {
            Self::new(932)
        }
    }

    /// The code page in use.
    pub fn code_page(&self) -> u32 {
        self.code_page
    }

    /// The Japanese-side key for a local filename key, or `None` when the flag
    /// is a no-op (code page 932, or the conversion leaves the name unchanged),
    /// the code page has no supported conversion, or a character cannot be
    /// encoded in it.
    pub fn japanese_key(&self, key: &[u16]) -> Option<Vec<u16>> {
        if self.code_page == 932 {
            return None;
        }
        let encoding = code_page_encoding(self.code_page)?;
        let local = String::from_utf16(key).ok()?;
        let bytes = encode_local(encoding, &local)?;
        let (japanese, _) = encoding_rs::SHIFT_JIS.decode_without_bom_handling(&bytes);
        let japanese: Vec<u16> = japanese.encode_utf16().collect();
        if japanese == key {
            None
        } else {
            Some(japanese)
        }
    }
}

/// The entry keys a wav needs in its folder's `desc.mrq`: the local filename
/// first, then — with sharing enabled and the code page conversion changing the
/// name — the Japanese-side key. With the flag on, *all* returned keys must
/// exist for the wav to count as having a table (#10 amendment).
pub fn entry_keys(file_name: &OsStr, sharing: Option<&Sharing>) -> Vec<Vec<u16>> {
    let local = filename_key(file_name);
    debug_assert!(!local.is_empty());
    let mut keys = vec![local];
    if let Some(sharing) = sharing
        && let Some(japanese) = sharing.japanese_key(&keys[0])
    {
        keys.push(japanese);
    }
    keys
}

/// The UTF-16 code units of a filename, verbatim on Windows and lossy
/// (UTF-8 → UTF-16) elsewhere, which is how mrq keys entries.
#[cfg(windows)]
pub fn filename_key(file_name: &OsStr) -> Vec<u16> {
    use std::os::windows::ffi::OsStrExt;

    file_name.encode_wide().collect()
}

/// The UTF-16 code units of a filename, verbatim on Windows and lossy
/// (UTF-8 → UTF-16) elsewhere, which is how mrq keys entries.
#[cfg(not(windows))]
pub fn filename_key(file_name: &OsStr) -> Vec<u16> {
    file_name.to_string_lossy().encode_utf16().collect()
}

#[cfg(windows)]
unsafe extern "system" {
    fn GetACP() -> u32;
}

/// The WHATWG encoding that stands in for a Windows ANSI code page, or `None`
/// for code pages without one (OEM and Mac code pages, mostly).
fn code_page_encoding(code_page: u32) -> Option<&'static encoding_rs::Encoding> {
    use encoding_rs::{
        BIG5, EUC_KR, GBK, IBM866, KOI8_R, KOI8_U, SHIFT_JIS, UTF_8, WINDOWS_874, WINDOWS_1250,
        WINDOWS_1251, WINDOWS_1252, WINDOWS_1253, WINDOWS_1254, WINDOWS_1255, WINDOWS_1256,
        WINDOWS_1257, WINDOWS_1258,
    };

    Some(match code_page {
        866 => IBM866,
        874 => WINDOWS_874,
        932 => SHIFT_JIS,
        936 => GBK,
        949 => EUC_KR,
        950 => BIG5,
        1250 => WINDOWS_1250,
        1251 => WINDOWS_1251,
        1252 => WINDOWS_1252,
        1253 => WINDOWS_1253,
        1254 => WINDOWS_1254,
        1255 => WINDOWS_1255,
        1256 => WINDOWS_1256,
        1257 => WINDOWS_1257,
        1258 => WINDOWS_1258,
        20866 => KOI8_R,
        21866 => KOI8_U,
        65001 => UTF_8,
        _ => return None,
    })
}

/// Encode `text` with the code page; `None` when any character has no mapping
/// there (there would be nothing sensible to key a shared entry on).
fn encode_local(encoding: &'static encoding_rs::Encoding, text: &str) -> Option<Vec<u8>> {
    let (bytes, _, had_errors) = encoding.encode(text);
    if had_errors {
        None
    } else {
        Some(bytes.into_owned())
    }
}
