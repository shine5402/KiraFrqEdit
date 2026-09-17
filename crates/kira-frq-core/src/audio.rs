//! WAV decode and normalization for the generation pipeline: decode PCM via
//! hound, downmix to mono, resample to 44.1 kHz via rubato (#5; research in
//! `docs/research/wav-decode-resample.md`).
//!
//! Input is normalized to the mono 44.1 kHz timeline the table formats assume;
//! non-44.1 kHz, non-16-bit and non-mono input is processed best-effort and
//! warned per file (#11). WORLD stays in `kira-frq-world` and the table formats
//! in their modules; this one hands the pipeline plain samples.

use std::io::{Read, Seek};
use std::path::Path;

/// Sample rate every UTAU table format implicitly assumes.
const TARGET_RATE_HZ: u32 = 44_100;

/// Chunk hint for [`rubato::Fft::new`] (the offline-batch settings from #11).
const RESAMPLER_CHUNK_HINT: usize = 2048;

/// The voicebank format the consuming tooling expects (#11).
const EXPECTED_BITS_PER_SAMPLE: u16 = 16;

/// One wav, decoded to mono 44.1 kHz samples plus the per-file warnings the
/// run report carries (#11).
#[derive(Debug, Clone, PartialEq)]
pub struct DecodedAudio {
    pub samples: Vec<f64>,
    /// Non-fatal deviations from the expected 44.1 kHz/16-bit/mono input;
    /// empty for a normal voicebank wav.
    pub warnings: Vec<String>,
}

/// Per-file decode result for the pipeline's bookkeeping.
#[derive(Debug, Clone, PartialEq)]
pub enum DecodeOutcome {
    /// Zero frames: the run skips it and analyzes nothing (#11).
    Empty,
    /// Decoded, downmixed and normalized to mono 44.1 kHz.
    Decoded(DecodedAudio),
    /// Undecodable (non-PCM, malformed, truncated, ...); the reason goes into
    /// the run report and partial samples are never returned.
    Failed(String),
}

/// Decode `path` into mono 44.1 kHz samples, or report why it cannot be used.
pub fn decode_wav(path: &Path) -> DecodeOutcome {
    let mut reader = match hound::WavReader::open(path) {
        Ok(reader) => reader,
        Err(error) => return DecodeOutcome::Failed(decode_error(&error)),
    };
    let spec = reader.spec();
    if reader.duration() == 0 {
        return DecodeOutcome::Empty;
    }

    let samples = match read_samples(&mut reader, &spec) {
        Ok(samples) => samples,
        Err(error) => return DecodeOutcome::Failed(decode_error(&error)),
    };

    let channels = usize::from(spec.channels);
    let mut warnings = Vec::new();
    if spec.sample_rate != TARGET_RATE_HZ {
        warnings.push(format!(
            "resampled from {} Hz to {TARGET_RATE_HZ} Hz",
            spec.sample_rate
        ));
    }
    if spec.bits_per_sample != EXPECTED_BITS_PER_SAMPLE {
        warnings.push(format!(
            "{}-bit input; {EXPECTED_BITS_PER_SAMPLE}-bit is the expected voicebank format",
            spec.bits_per_sample
        ));
    }
    if channels != 1 {
        warnings.push(format!("averaged {channels} channels to mono"));
    }

    let samples = downmix(samples, channels);
    let samples = if spec.sample_rate == TARGET_RATE_HZ {
        samples
    } else {
        match resample_to_target(&samples, spec.sample_rate) {
            Ok(samples) => samples,
            Err(reason) => return DecodeOutcome::Failed(reason),
        }
    };

    DecodeOutcome::Decoded(DecodedAudio { samples, warnings })
}

/// Read every sample as f64: ints scale by `2^(bits-1)`, float32 is used
/// as-is without clamping (#11). An error mid-iteration (truncated or
/// unreadable file) fails the whole decode.
fn read_samples<R: Read + Seek>(
    reader: &mut hound::WavReader<R>,
    spec: &hound::WavSpec,
) -> Result<Vec<f64>, hound::Error> {
    match spec.sample_format {
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1u64 << (spec.bits_per_sample - 1)) as f64;
            reader
                .samples::<i32>()
                .map(|sample| sample.map(|value| f64::from(value) * scale))
                .collect()
        }
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .map(|sample| sample.map(f64::from))
            .collect(),
    }
}

/// The pipeline's failure reason; non-PCM gets the research's actionable hint.
fn decode_error(error: &hound::Error) -> String {
    match error {
        hound::Error::Unsupported => {
            "unsupported WAV encoding (non-PCM); convert to PCM first".to_string()
        }
        other => other.to_string(),
    }
}

/// Fold interleaved frames to their channel mean; a partial trailing frame is
/// dropped rather than averaged short.
fn downmix(samples: Vec<f64>, channels: usize) -> Vec<f64> {
    if channels == 1 {
        return samples;
    }
    samples
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f64>() / channels as f64)
        .collect()
}

fn resample_to_target(samples: &[f64], input_rate: u32) -> Result<Vec<f64>, String> {
    use rubato::Resampler;
    use rubato::audioadapter_buffers::direct::InterleavedSlice;

    let failure = |error: &dyn std::fmt::Display| {
        format!("cannot resample {input_rate} Hz to {TARGET_RATE_HZ} Hz: {error}")
    };

    let mut resampler = rubato::Fft::<f64>::new(
        input_rate as usize,
        TARGET_RATE_HZ as usize,
        RESAMPLER_CHUNK_HINT,
        1,
        rubato::FixedSync::Both,
    )
    .map_err(|error| failure(&error))?;

    // `process_all` only trims its startup delay after a full internal chunk;
    // for clips of at most `input_frames_next()` frames it would return the
    // untrimmed warm-up. Pad those past one chunk so the trim runs, then cut
    // the zero-padded tail back to the input length's output count (#11).
    let original_len = samples.len();
    let padded_len = original_len.max(resampler.input_frames_next() + 1);
    let mut padded = samples.to_vec();
    padded.resize(padded_len, 0.0);

    let input = InterleavedSlice::new(&padded, 1, padded_len).map_err(|error| failure(&error))?;
    let output = resampler
        .process_all(&input, padded_len, None)
        .map_err(|error| failure(&error))?;

    let ratio = f64::from(TARGET_RATE_HZ) / f64::from(input_rate);
    let mut samples = output.take_data();
    samples.truncate((ratio * original_len as f64).ceil() as usize);
    Ok(samples)
}
