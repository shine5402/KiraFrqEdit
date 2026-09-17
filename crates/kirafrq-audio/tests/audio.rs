//! Contract for `kirafrq_audio` — #19's task list and #11's WAV input
//! policy, exercised on real files in a scratch folder.
//!
//! The alignment tests guard the pad workaround in `resample_to_target`; see
//! its comment for why the pad is load-bearing.

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use hound::{SampleFormat, WavSpec, WavWriter};
use kirafrq_audio::{DecodeOutcome, DecodedAudio, decode_wav};

/// A scratch folder that deletes itself.
struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "frq-audio-{}-{}-{}",
            tag,
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn join(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn spec(channels: u16, sample_rate: u32, bits_per_sample: u16, format: SampleFormat) -> WavSpec {
    WavSpec {
        channels,
        sample_rate,
        bits_per_sample,
        sample_format: format,
    }
}

fn write_wav<S: hound::Sample + Copy>(
    scratch: &Scratch,
    name: &str,
    spec: WavSpec,
    samples: &[S],
) -> PathBuf {
    let path = scratch.join(name);
    let mut writer = WavWriter::create(&path, spec).unwrap();
    for sample in samples {
        writer.write_sample(*sample).unwrap();
    }
    writer.finalize().unwrap();
    path
}

fn decoded(outcome: DecodeOutcome) -> DecodedAudio {
    match outcome {
        DecodeOutcome::Decoded(audio) => audio,
        other => panic!("expected Decoded, got {other:?}"),
    }
}

fn assert_close(got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "got {got:?}, want {want:?}");
    for (index, (got, want)) in got.iter().zip(want).enumerate() {
        assert!(
            (got - want).abs() < 1e-9,
            "sample {index}: got {got}, want {want}"
        );
    }
}

#[test]
fn decodes_16_bit_44100_mono_without_warnings() {
    let scratch = Scratch::new("pcm16");
    let source: Vec<i16> = vec![0, 16_384, -8_192, 32_767, -32_768];
    let path = write_wav(
        &scratch,
        "a.wav",
        spec(1, 44_100, 16, SampleFormat::Int),
        &source,
    );

    let audio = decoded(decode_wav(&path));
    assert!(audio.warnings.is_empty(), "{:?}", audio.warnings);
    let want: Vec<f64> = source.iter().map(|v| f64::from(*v) / 32_768.0).collect();
    assert_close(&audio.samples, &want);
}

#[test]
fn decodes_8_bit_with_bias_correction_and_scale() {
    let scratch = Scratch::new("pcm8");
    let source: Vec<i8> = vec![0, 127, -128, 64];
    let path = write_wav(
        &scratch,
        "a.wav",
        spec(1, 44_100, 8, SampleFormat::Int),
        &source,
    );

    let audio = decoded(decode_wav(&path));
    assert_eq!(
        audio.warnings,
        ["8-bit input; 16-bit is the expected voicebank format"]
    );
    assert_close(&audio.samples, &[0.0, 127.0 / 128.0, -1.0, 0.5]);
}

#[test]
fn decodes_24_bit_with_scale() {
    let scratch = Scratch::new("pcm24");
    let source: Vec<i32> = vec![8_388_607, -8_388_608, 0, 4_194_304];
    let path = write_wav(
        &scratch,
        "a.wav",
        spec(1, 44_100, 24, SampleFormat::Int),
        &source,
    );

    let audio = decoded(decode_wav(&path));
    assert_eq!(
        audio.warnings,
        ["24-bit input; 16-bit is the expected voicebank format"]
    );
    assert_close(&audio.samples, &[8_388_607.0 / 8_388_608.0, -1.0, 0.0, 0.5]);
}

#[test]
fn decodes_32_bit_int_with_scale() {
    let scratch = Scratch::new("pcm32");
    let source: Vec<i32> = vec![i32::MAX, i32::MIN, 0, 1_073_741_824];
    let path = write_wav(
        &scratch,
        "a.wav",
        spec(1, 44_100, 32, SampleFormat::Int),
        &source,
    );

    let audio = decoded(decode_wav(&path));
    assert_eq!(
        audio.warnings,
        ["32-bit input; 16-bit is the expected voicebank format"]
    );
    assert_close(
        &audio.samples,
        &[f64::from(i32::MAX) / 2_147_483_648.0, -1.0, 0.0, 0.5],
    );
}

#[test]
fn decodes_float32_as_is_without_clamping() {
    let scratch = Scratch::new("float32");
    let source: Vec<f32> = vec![0.5, -0.25, 1.5, -2.0];
    let path = write_wav(
        &scratch,
        "a.wav",
        spec(1, 44_100, 32, SampleFormat::Float),
        &source,
    );

    let audio = decoded(decode_wav(&path));
    assert_eq!(
        audio.warnings,
        ["32-bit input; 16-bit is the expected voicebank format"]
    );
    assert_close(&audio.samples, &[0.5, -0.25, 1.5, -2.0]);
}

#[test]
fn averages_multiple_channels_to_mono() {
    let scratch = Scratch::new("stereo");
    let interleaved: Vec<i16> = vec![16_384, 8_192, -16_384, 0, 32_767, -32_768];
    let path = write_wav(
        &scratch,
        "a.wav",
        spec(2, 44_100, 16, SampleFormat::Int),
        &interleaved,
    );

    let audio = decoded(decode_wav(&path));
    assert_eq!(audio.warnings, ["averaged 2 channels to mono"]);
    assert_close(
        &audio.samples,
        &[
            (16_384.0 + 8_192.0) / 2.0 / 32_768.0,
            (-16_384.0 + 0.0) / 2.0 / 32_768.0,
            (32_767.0 + -32_768.0) / 2.0 / 32_768.0,
        ],
    );
}

#[test]
fn warns_about_every_deviation_from_16_bit_44100_mono() {
    let scratch = Scratch::new("warn");
    let source: Vec<i32> = (0..200).map(|i| i * 1_000).collect();
    let path = write_wav(
        &scratch,
        "a.wav",
        spec(2, 48_000, 24, SampleFormat::Int),
        &source,
    );

    let audio = decoded(decode_wav(&path));
    assert_eq!(
        audio.warnings,
        [
            "resampled from 48000 Hz to 44100 Hz",
            "24-bit input; 16-bit is the expected voicebank format",
            "averaged 2 channels to mono",
        ]
    );
    assert_eq!(
        audio.samples.len(),
        (147.0 * 100.0 / 160.0f64).ceil() as usize
    );
}

#[test]
fn resamples_48_khz_to_44100_with_the_canonical_length() {
    let scratch = Scratch::new("48k");
    let source: Vec<f32> = vec![0.0; 1000];
    let path = write_wav(
        &scratch,
        "a.wav",
        spec(1, 48_000, 32, SampleFormat::Float),
        &source,
    );

    let audio = decoded(decode_wav(&path));
    assert_eq!(
        audio.warnings,
        [
            "resampled from 48000 Hz to 44100 Hz",
            "32-bit input; 16-bit is the expected voicebank format",
        ]
    );
    assert_eq!(audio.samples.len(), 919);
}

/// A Hann-windowed sine burst inside a clip shorter than one resampler chunk.
#[test]
fn sub_chunk_48_khz_clip_stays_time_aligned() {
    let scratch = Scratch::new("subchunk");
    const LEN: usize = 1000;
    const START: usize = 400;
    const END: usize = 600;
    let source = burst(LEN, START, END);
    let path = write_wav(
        &scratch,
        "a.wav",
        spec(1, 48_000, 32, SampleFormat::Float),
        &source,
    );

    let audio = decoded(decode_wav(&path));
    assert_eq!(
        audio.warnings,
        [
            "resampled from 48000 Hz to 44100 Hz",
            "32-bit input; 16-bit is the expected voicebank format",
        ]
    );
    assert_eq!(audio.samples.len(), 919);

    // The 200-frame burst is centred on frame 500 at 48 kHz, i.e. on output
    // frame 459.4 after the 147/160 ratio; the energy centroid is unchanged by
    // the resampler's linear-phase filter.
    let centroid = energy_centroid(&audio.samples).expect("burst energy");
    assert!(
        (centroid - 459.4).abs() < 25.0,
        "burst centroid {centroid}, want about 459.4 (untuned delay would push it past the clip)"
    );
}

#[test]
fn long_48_khz_clip_stays_time_aligned() {
    let scratch = Scratch::new("long");
    const LEN: usize = 6000;
    const START: usize = 2900;
    const END: usize = 3100;
    let source = burst(LEN, START, END);
    let path = write_wav(
        &scratch,
        "a.wav",
        spec(1, 48_000, 32, SampleFormat::Float),
        &source,
    );

    let audio = decoded(decode_wav(&path));
    assert_eq!(
        audio.warnings,
        [
            "resampled from 48000 Hz to 44100 Hz",
            "32-bit input; 16-bit is the expected voicebank format",
        ]
    );
    assert_eq!(
        audio.samples.len(),
        (147.0 * LEN as f64 / 160.0).ceil() as usize
    );

    let centroid = energy_centroid(&audio.samples).expect("burst energy");
    assert!(
        (centroid - 2756.25).abs() < 25.0,
        "burst centroid {centroid}, want about 2756.25"
    );
}

fn burst(len: usize, start: usize, end: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; len];
    let width = (end - start) as f64;
    for (index, sample) in out.iter_mut().enumerate().take(end).skip(start) {
        let position = (index - start) as f64;
        let window = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * position / width).cos();
        let tone = (2.0 * std::f64::consts::PI * 440.0 * position / 48_000.0).sin();
        *sample = (0.8 * window * tone) as f32;
    }
    out
}

fn energy_centroid(samples: &[f64]) -> Option<f64> {
    let total: f64 = samples.iter().map(|s| s * s).sum();
    if total == 0.0 {
        return None;
    }
    let weighted: f64 = samples
        .iter()
        .enumerate()
        .map(|(index, s)| index as f64 * s * s)
        .sum();
    Some(weighted / total)
}

#[test]
fn zero_frames_is_empty_for_the_pipeline_skip() {
    let scratch = Scratch::new("empty");
    let path = write_wav(
        &scratch,
        "a.wav",
        spec(1, 44_100, 16, SampleFormat::Int),
        &[] as &[i16],
    );

    assert_eq!(decode_wav(&path), DecodeOutcome::Empty);
}

#[test]
fn truncated_wav_fails_without_partial_samples() {
    let scratch = Scratch::new("truncated");
    let source: Vec<i16> = (0..1000).collect();
    let path = write_wav(
        &scratch,
        "a.wav",
        spec(1, 44_100, 16, SampleFormat::Int),
        &source,
    );
    let file = fs::OpenOptions::new().write(true).open(&path).unwrap();
    file.set_len(44 + 200).unwrap();

    let outcome = decode_wav(&path);
    assert!(matches!(outcome, DecodeOutcome::Failed(_)), "{outcome:?}");
}

#[test]
fn non_pcm_wav_fails_with_a_reason() {
    let scratch = Scratch::new("adpcm");
    let source: Vec<i16> = vec![0; 100];
    let path = write_wav(
        &scratch,
        "a.wav",
        spec(1, 44_100, 16, SampleFormat::Int),
        &source,
    );
    // Patch the fmt chunk's format tag from PCM (1) to ADPCM (2); the fmt chunk
    // payload starts 20 bytes in (RIFF header + "fmt " + size).
    let mut bytes = fs::read(&path).unwrap();
    bytes[20..22].copy_from_slice(&2u16.to_le_bytes());
    fs::write(&path, bytes).unwrap();

    match decode_wav(&path) {
        DecodeOutcome::Failed(reason) => {
            assert!(!reason.is_empty(), "failure needs a reason");
        }
        other => panic!("expected Failed, got {other:?}"),
    }
}

#[test]
fn garbage_file_fails_with_a_reason() {
    let scratch = Scratch::new("garbage");
    let path = scratch.join("a.wav");
    fs::write(&path, b"RIFF\x00\x00\x00\x00WAVEnot-really").unwrap();

    match decode_wav(&path) {
        DecodeOutcome::Failed(reason) => assert!(!reason.is_empty(), "failure needs a reason"),
        other => panic!("expected Failed, got {other:?}"),
    }
}

#[test]
fn missing_file_fails_with_a_reason() {
    let scratch = Scratch::new("missing");
    let path = scratch.join("nope.wav");

    match decode_wav(&path) {
        DecodeOutcome::Failed(reason) => assert!(!reason.is_empty(), "failure needs a reason"),
        other => panic!("expected Failed, got {other:?}"),
    }
}
