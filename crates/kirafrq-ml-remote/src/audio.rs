//! The request body: the wav's decoded mono 44.1 kHz samples, sent exactly.
//! Nothing is resampled or rounded on the wire, so the server's model input
//! is bit-identical to a local run's.
//!
//! The smallest exact encoding wins: 16-bit FLAC when every sample sits on
//! the 16-bit grid (a normal UTAU wav), 24-bit FLAC on the 24-bit grid (a
//! 24-bit source, or a 16-bit stereo mixdown), else raw little-endian f64
//! (float sources, or a stray rate resampled on decode). FLAC is pure Rust
//! both ways (`flacenc` encodes, `claxon` decodes), so no C build.

use crate::SAMPLE_RATE;

/// The body's `Content-Type` for FLAC.
pub(crate) const FLAC: &str = "audio/flac";
/// The body's `Content-Type` for raw little-endian f64 samples.
pub(crate) const F64LE: &str = "application/octet-stream";

/// Encode `samples` exactly, returning the `Content-Type` and the body.
pub(crate) fn encode(samples: &[f64]) -> Result<(&'static str, Vec<u8>), String> {
    for bits in [16, 24] {
        if let Some(integers) = on_grid(samples, bits) {
            return Ok((FLAC, flac(&integers, bits)?));
        }
    }
    Ok((
        F64LE,
        samples
            .iter()
            .flat_map(|sample| sample.to_le_bytes())
            .collect(),
    ))
}

/// `samples` as `bits`-bit integers, if every one is exactly representable;
/// the WAV decode scales integers by `2^(bits-1)`, so this inverts it.
fn on_grid(samples: &[f64], bits: u32) -> Option<Vec<i32>> {
    let scale = f64::from(1_u32 << (bits - 1));
    samples
        .iter()
        .map(|&sample| {
            let value = sample * scale;
            (value.fract() == 0.0 && value >= -scale && value < scale).then_some(value as i32)
        })
        .collect()
}

fn flac(samples: &[i32], bits: u32) -> Result<Vec<u8>, String> {
    use flacenc::component::BitRepr;
    use flacenc::error::Verify;

    let config = flacenc::config::Encoder::default()
        .into_verified()
        .map_err(|(_, error)| format!("FLAC encoder config: {error:?}"))?;
    let source =
        flacenc::source::MemSource::from_samples(samples, 1, bits as usize, SAMPLE_RATE as usize);
    let stream = flacenc::encode_with_fixed_block_size(&config, source, config.block_size)
        .map_err(|error| format!("FLAC encode failed: {error:?}"))?;
    let mut sink = flacenc::bitsink::ByteSink::new();
    stream
        .write(&mut sink)
        .map_err(|error| format!("FLAC write failed: {error:?}"))?;
    Ok(sink.as_slice().to_vec())
}

/// Decode a request body of `content_type` back to the samples the client
/// encoded; FLAC must be 44.1 kHz mono at 16 or 24 bits.
#[cfg(feature = "server")]
pub(crate) fn decode(content_type: &str, bytes: &[u8]) -> Result<Vec<f64>, String> {
    match content_type {
        FLAC => {
            let mut reader = claxon::FlacReader::new(std::io::Cursor::new(bytes))
                .map_err(|error| format!("the body is not FLAC: {error}"))?;
            let info = reader.streaminfo();
            if info.sample_rate != SAMPLE_RATE
                || info.channels != 1
                || !matches!(info.bits_per_sample, 16 | 24)
            {
                return Err(format!(
                    "the audio must be {SAMPLE_RATE} Hz mono FLAC at 16 or 24 bits, not {} Hz, \
                     {}-bit, {} channel(s)",
                    info.sample_rate, info.bits_per_sample, info.channels
                ));
            }
            let scale = f64::from(1_u32 << (info.bits_per_sample - 1));
            reader
                .samples()
                .map(|sample| {
                    sample
                        .map(|value| f64::from(value) / scale)
                        .map_err(|error| format!("the FLAC stream is corrupt: {error}"))
                })
                .collect()
        }
        F64LE => {
            if !bytes.len().is_multiple_of(8) {
                return Err(format!(
                    "{} bytes is not a whole number of f64 samples",
                    bytes.len()
                ));
            }
            Ok(bytes
                .chunks_exact(8)
                .map(|chunk| f64::from_le_bytes(chunk.try_into().expect("8 bytes")))
                .collect())
        }
        other => Err(format!(
            "unsupported Content-Type `{other}`; expected `{FLAC}` or `{F64LE}`"
        )),
    }
}

#[cfg(all(test, feature = "server"))]
mod tests {
    use super::*;

    fn round_trip(samples: &[f64]) -> (&'static str, Vec<f64>) {
        let (content_type, body) = encode(samples).unwrap();
        (content_type, decode(content_type, &body).unwrap())
    }

    fn tone(len: usize, scale: f64) -> Vec<f64> {
        (0..len)
            .map(|index| ((index as f64 * 0.01).sin() * 0.7 * scale).round() / scale)
            .collect()
    }

    #[test]
    fn a_16_bit_wav_goes_as_16_bit_flac_and_comes_back_exact() {
        let samples: Vec<f64> = tone(5_000, 32_768.0)
            .into_iter()
            .chain([-1.0, 32_767.0 / 32_768.0, 0.0])
            .collect();
        let (content_type, decoded) = round_trip(&samples);
        assert_eq!(content_type, FLAC);
        assert_eq!(decoded, samples);
        let (_, body) = encode(&samples).unwrap();
        assert!(
            body.len() < samples.len() * 2,
            "FLAC compresses: {}",
            body.len()
        );
    }

    #[test]
    fn a_24_bit_grid_goes_as_24_bit_flac_and_comes_back_exact() {
        // A 16-bit stereo mixdown lands on the 17-bit grid.
        let samples = tone(5_000, 65_536.0);
        let (content_type, decoded) = round_trip(&samples);
        assert_eq!(content_type, FLAC);
        assert_eq!(decoded, samples);
    }

    #[test]
    fn anything_else_goes_raw_and_comes_back_exact() {
        let samples: Vec<f64> = (0..1_000)
            .map(|index| (index as f64 * 0.013).sin() * 0.3)
            .chain([1.5, -1.0])
            .collect();
        let (content_type, decoded) = round_trip(&samples);
        assert_eq!(content_type, F64LE);
        assert_eq!(decoded, samples);
    }

    #[test]
    fn a_bad_body_is_rejected_with_a_reason() {
        assert!(decode(FLAC, b"not flac").is_err());
        assert!(decode(F64LE, &[0; 7]).is_err());
        assert!(decode("text/plain", &[]).is_err());
    }
}
