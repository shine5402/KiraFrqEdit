# Neutral `FrequencyTable` as the shared model; native formats are IO adapters

Status: accepted

All frequency-table formats share one in-memory model — `kira_frq_core::FrequencyTable`
(sample rate, hop, f0 per frame, optional amplitude, key frequency) — and format modules read
and write it. Generation builds the neutral table directly from the estimator's track;
conversions between formats go through it and are documented lossy; the future editor and
check/repair milestones work on it. Native format types survive only where byte-faithful
preservation demands them (mrq merge must keep untouched entries verbatim; a pmk rewrite must
preserve stored `pos_end` values).

## Considered options

- **Per-format tables only** — each writer projects straight off the estimator track. Exact by
  construction, but the editor and format conversion would need an N×N web of bridges, and no
  type would carry the domain's central concept.
- **One pre-projected hop-grid table as *the* model** — would force pmk's period-segment walk
  into a hop grid it does not have. The neutral model is an interchange/working model, not an
  encoding.

## Consequences

- Conversion cannot be byte-exact where formats quantise differently (pmk period codes, mrq
  `float32` f0, frq amplitude); this is expected and matches frqeditor's documented behaviour.
- `kira-frq-core` takes no position on how f0 was estimated, so it stays free of the
  WORLD/FFI dependency; the editor milestone does not link WORLD.
- The table's `hop_samples` is the source of truth for the estimator grid too: the pipeline
  derives WORLD's `frame_period` from it (≈ 5.805 ms at 44.1 kHz/256) instead of using the 5 ms
  default, so estimator frames land exactly on table frames.
