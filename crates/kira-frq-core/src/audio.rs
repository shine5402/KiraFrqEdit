//! WAV decode and normalization for the generation pipeline: PCM decode via
//! hound, downmix to mono, resample to 44.1 kHz via rubato (#5; research in
//! `docs/research/wav-decode-resample.md`).
//!
//! Input policy (zero-length wavs, sidecar naming) is #11's call; the decode
//! implementation lands with the pipeline ticket.
