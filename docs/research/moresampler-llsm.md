# moresampler `.llsm` caches and `desc.mrq` staleness

Status: **verified** by black-box testing of the installed moresampler **0.8.4**
(`engines_original\moresampler.exe`) plus the shipped 0.8.3 readme. Recommendations at the end are
clearly separated from observed behavior.

This document answers: what `.llsm` files are, when moresampler trusts or rebuilds them, which
`timestamp`/`modified` values a third-party `desc.mrq` writer must use, and how KiraFrqGen should
invalidate caches. Format details of `desc.mrq` itself are in [mrq-format.md](mrq-format.md).

## What `.llsm` files are

- **One cache per source wav, in the wav's folder**, named by appending `.llsm` to the *full* wav
  filename: `_かきくけこ.wav` → `_かきくけこ.wav.llsm` (across several pitch folders in the
  local corpus). Not one per folder.
- They are moresampler's serialized LLSM/LMIX analysis of the sample (libllsm layer-0 model:
  harmonic + noise parameters, with a few metadata fields such as `version` and `duration`).
  There is no public spec; the classic moresampler is the writer. The `version` field is a float at
  byte offset 19 in 0.8.4 output and reads `524.0`.
- **They bake f0 in.** Verified: changing only the `desc.mrq` f0 of a wav and deleting its `.llsm`
  changes the regenerated `.llsm` bytes and the rendered audio (constant 440 Hz → `.llsm`
  453,594 → 382,104 bytes for the test sample; all-unvoiced → 180,469 bytes, output RMS
  2098 → 488).
- They are written automatically in the analysis stage of a render. A plain UTAU 13-argument
  resampler call (`moresampler.exe in.wav out.wav A2 100 "" 0 1000 0 0 100 0 120 pb.txt`) on a
  folder with no caches wrote both `desc.mrq` and `in.wav.llsm` next to the input wav.

## When moresampler distrusts the `.llsm` ("data record")

Decision branches in 0.8.4 (message strings extracted from the binary; the mtime and version rows
were also reproduced end-to-end in the log via `dump-log-file`, the invalid rows are strings only):

| Condition | Log message | Action |
| --- | --- | --- |
| no `.llsm` | "Data record is not detected. The required file will be generated in the current session." | analyze; write `.llsm` (+ a `desc.mrq` entry if absent) |
| unreadable/invalid | "An invalid data record is detected. This file will be re-generated in the current session." | re-analyze |
| `wav_mtime > llsm_mtime` (whole seconds) | "The .wav file is newer than the data record. The record will be updated in the current session." | re-analyze, rewrite `.llsm` |
| `.llsm` version `< 520` | "A data record with older and incompatible version is detected. This record will be updated in the current session." | re-analyze, rewrite as version 524 |
| `.llsm` version 520..523 | "A data record with older version is detected. You may consider delete the data record and moresampler will automatically re-analyze by next session." | **load as-is** (file untouched, warning only) |
| `.llsm` version 524 | (none) | load |
| `.llsm` version 525..599 | "A data record with newer version is detected. You may consider update moresampler." | **load as-is** (file untouched) |
| `.llsm` version `>= 600` | "A data record with newer but incompatible version is detected. Synthesis will terminate for overwrite-protection. ..." | exit code 1, no output wav, file untouched; only deleting the cache recovers |

Boundary details, measured:

- The mtime comparison is **strict and truncated to whole seconds**: wav mtime `T` vs llsm mtime `T`
  is fresh; wav `T+0.9s` vs llsm `T` is also fresh; only a wav mtime greater than the llsm's (in
  whole seconds) makes it stale — the `.llsm`'s content, `desc.mrq`, and config changes are
  irrelevant.
- 0.8.4's compatibility window for loading is `[520, 599]`; the higher bound `600` is fatal.
  0.8.4 writes version `524`; the 0.8.3 changelog is the source for the option that governs this:
  `auto-update-llsm-mrq` (default on).
- With `auto-update-llsm-mrq off`, the wav-newer check is simply skipped — a stale `.llsm` is used
  silently. A **missing** `.llsm` is still generated. So cache deletion forces regeneration even
  for users who turned the option off.
- Analysis settings are not fingerprinted: changing e.g. `analysis-f0-min` with a fresh `.llsm`
  present does not trigger re-analysis (verified).

## When moresampler trusts a `desc.mrq` entry

A `.llsm` is only consulted against `desc.mrq` when it must (re)analyze. The entry for the bare wav
filename is then trusted or rejected as follows (0.8.4; the `timestamp`/`modified` rows were
reproduced end-to-end, the mrq file-version row is a binary string):

| Entry state | Log message | Result |
| --- | --- | --- |
| no entry | "A new mrq data entry is created." | estimate f0, write entry (`timestamp = now`, `modified = 524`) |
| `timestamp == 0` | "A mrq data entry is found and loaded." | trusted, entry untouched |
| `timestamp < wav_mtime` (whole seconds) | "A mrq data entry is detected but it is older than the .wav file. Moresampler will update the entry in current session." | re-estimate f0, rewrite entry |
| `timestamp == wav_mtime` (whole seconds) | "found and loaded" | trusted |
| `modified != 0 && modified < 520` | "A mrq data entry is detected but it is generated by a deprecated version of Moresampler. ..." | re-estimate f0, rewrite entry |
| otherwise (`modified == 0`, or `modified >= 520`) | "found and loaded" | trusted |
| mrq file version != 2 | "A mrq data descriptor of unsupported version is detected. The file is ignored." | ignored (f0 estimated) |

Measured boundaries: `modified=519` is deprecated, `modified=520` is accepted; **there is no upper
bound on `modified` in 0.8.4** (600 and 1000 were trusted, entry untouched) — the `520..599`
window is the `.llsm` *version* field, not `modified`. Sub-second wav mtimes are ignored: wav
`T+0.9s` with entry `timestamp = T` stayed trusted. `mrq.h` documents `modified` as "modified by
hand (0: true, >=1: false)", i.e. `0` is the hand-edit marker; `mrq.c` always stamps `timestamp`
with `time()` when writing v2, so the caller controls only `modified`.

`.frq` participates only under the opt-in `load-frq` (default `off`): `strict` loads `.frq` pitch
when `desc.mrq` is unavailable, `on` uses it to correct moresampler's own estimate, and the result
is written to `desc.mrq` either way.

## The drift problem: a fresh `.llsm` hides `desc.mrq`

Two decisive experiments:

1. Fresh `.llsm` (mtime newer than wav), `desc.mrq` entry rewritten with `f0 = 440 Hz` constant and
   `timestamp = now + 1000` → moresampler logged nothing about mrq, left the entry alone, and
   produced byte-identical output to the unmodified cache. **A fresh `.llsm` wins; edited f0 is
   silently ignored.**
2. Delete the `.llsm`, keep the same entry → "A mrq data entry is found and loaded.", the cache is
   rebuilt from the 440 Hz contour, and the output changes accordingly.

Consequence: after KiraFrqGen writes new f0 for a wav, **editing `desc.mrq` alone has no effect**
until the `.llsm` is removed (or the wav itself is rewritten with a newer mtime). Deleting the
`.llsm` does not lose the generated f0 — moresampler rebuilds it *from* the trusted `desc.mrq`
entry.

## Recommendations (for KiraFrqGen; not upstream behavior)

### `timestamp` / `modified` for written entries

- Write one v2 entry per generated wav, keyed by the bare wav filename, into the `desc.mrq` of the
  wav's own folder.
- `timestamp = max(now_seconds, floor(wav_mtime_seconds))`, computed **after** the wav file is
  written. Equality is enough to be trusted (the comparison is strict `<`), so `time(NULL)` after
  the wav write is normally sufficient; clamping to the wav's mtime also covers restored/copied
  timestamps and `auto-update` races.
- `modified = 0`.
- Rationale: `0` is the "hand-edited" marker and never hits the deprecated-version check, in 0.8.4
  or any version that keeps that semantic. Claiming a moresampler version (e.g. 524) is fragile —
  a future release can raise the deprecated floor (as the `.llsm` floor did) and invalidate
  everything we wrote. Never write `timestamp = 0` for generated tables: that means "trusted
  forever" and would mask later wav edits. Reserve `timestamp = 0, modified = 0` for an explicit
  "pin/hand-edited" user action.
- If a wav is re-written or re-processed after its entry was written, write the entry again
  afterwards; otherwise moresampler re-estimates and overwrites the f0.
- If the canonical `mrq.c` is used, it sets `timestamp = time()` itself; `modified` is the only
  field the caller controls.

### `.llsm` deletion policy

- **Delete `<wav>.llsm` for every wav whose `desc.mrq` f0 entry KiraFrqGen writes** (the `mrq`
  target), whether or not the wav bytes changed. `frq` writes do not invalidate: moresampler reads
  `.frq` only under the opt-in `load-frq`, and only while (re)analyzing. Deletion is the only
  reliable invalidation: a fresh `.llsm` ignores `desc.mrq`, and the deletion path works even with
  `auto-update-llsm-mrq off`. It also unblocks rendering when a newer moresampler left a `>= 600`
  `.llsm` behind.
- **Never "touch" the wav to invalidate.** Making the wav newer than the entry triggers
  re-estimation, which discards the generated f0.
- **Never delete `desc.mrq`** (that would discard the f0 we just generated). Leave unrelated wavs'
  caches alone; expose a separate "clear all caches" batch action if needed.
- Do the deletion as part of the generation operation, not while UTAU/moresampler is rendering
  (multi-process rendering may be reading caches concurrently).
- **CLI:** when not specified and stdout is a TTY, prompt (default: delete). When non-interactive,
  delete by default. Provide an opt-out such as `--keep-llsm` (or `--llsm=ask|delete|keep`), and
  print the deleted paths.
- **GUI:** a checkbox (default on), e.g. "Delete moresampler `.llsm` caches for generated files",
  with the drift rationale in its tooltip.
- Rationale: regeneration is automatic and cheap (about 1 s for a 5.5 s sample, using the trusted
  `desc.mrq` entry), the caches are derived data, and the alternative is a resampler that silently
  sings the old f0.

## Open uncertainties

- `.llsm` serialization is proprietary; only the `version` field and mtime semantics are known.
  Do not parse or patch these files — delete them.
- How moresampler treats entries written with a different `nhop`/`fs` than its own analysis grid
  (our tests used its own 44100/256). A WORLD-generated table with a 5 ms hop may be interpolated,
  but this was not tested.
- Concurrency/locking when deleting caches while multiple moresampler instances render (UTAU
  multiprocess); not tested.
- Other writers/readers of `.llsm` (oto-generation drag-and-drop mode, third-party tools) were not
  investigated.
- mrq file-level versions other than 2 (the "unsupported version, file ignored" path) were not
  exercised end to end.

## Sources

- Local moresampler 0.8.4 binary in `engines_original` (path in `AGENTS.local.md`; stdout banner
  "Moresampler 0.8.4"); config `...\Resampler\moreconfig.txt` / mirror
  `moreconfig.txt`.
- The Complete Moresampler Tutorial (Kanru Hua, April 2016, written for 0.7.1), archived 2018-08-13:
  <https://archive.ph/CTGaA> — `load-frq` semantics, and "`desc.mrq` file is accessed only once
  when Moresampler generates `.llsm` files, since `.llsm` file already completely describes the
  speech sample".
- The bundled Moresampler readme (0.8.3; path in `AGENTS.local.md`): 0.6.1 changelog defines
  `auto-update-llsm-mrq` ("If the .wav file is newer than the .llsm file, then reanalyze. If the
  .wav file is also newer than the mrq data entry, then re-estimate pitch before reanalyzing
  .llsm."); 0.3.0 notes that `desc.mrq` is scanned before creating `.llsm` files.
- <https://github.com/Sleepwalking/mrq> — `mrq.c`/`mrq.h` field semantics and write behavior.
- Mirror <https://github.com/hungrierr/Moresampler> — 0.8.3 readme, example `moreconfig.txt`.
- Corpus: `**\*.wav.llsm` naming/mtime evidence.
- Black-box procedure (reproducible): copy one corpus wav to a scratch folder (renamed, to control
  the mrq key), copy `moresampler.exe` and a `moreconfig.txt` (`resampler-compatibility on`,
  `auto-update-llsm-mrq on`, `dump-log-file <path>`) to a scratch engine dir, invoke the 13-argument
  resampler CLI, then parse `desc.mrq` and inspect `.llsm` bytes/mtimes and the log. 0.8.4's own
  messages above are the branch labels.
