//! Local verification probe (not a CI test): runs the bundled SwiftF0 over
//! real 44.1 kHz wavs through the provider's real seam and dumps each wav's
//! 16 kHz input (`.f32`, little-endian f32) plus the raw native-grid
//! `pitch_hz`/`confidence` (`.json`), so the Python `swift_f0` reference can
//! be diffed against them on identical audio.
//!
//! Run:
//! `cargo run -p kirafrq-ml-provider --features ort --example swiftf0_probe -- \
//!      <out_dir> <wav>...`

use std::fmt::Write as _;

use kirafrq_ml_provider::model::ModelSource;
use kirafrq_ml_provider::policy::UvPolicy;
use kirafrq_ml_provider::resample::resample_to_model_rate;
use kirafrq_ml_provider::swiftf0::{BUNDLED_MODEL, SwiftF0};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: swiftf0_probe <out_dir> <wav>...");
        std::process::exit(2);
    }
    let out = std::path::PathBuf::from(&args[1]);
    std::fs::create_dir_all(&out).expect("create out dir");
    let estimator = SwiftF0::new(
        ModelSource::Bundled {
            file_name: "swiftf0.onnx",
            bytes: BUNDLED_MODEL,
        },
        UvPolicy::swiftf0(None),
        1,
    );

    for (index, wav) in args[2..].iter().enumerate() {
        let decoded = match kirafrq_audio::decode_wav(std::path::Path::new(wav)) {
            kirafrq_audio::DecodeOutcome::Decoded(decoded) => decoded,
            kirafrq_audio::DecodeOutcome::Empty => {
                eprintln!("{wav}: empty wav");
                continue;
            }
            kirafrq_audio::DecodeOutcome::Failed(reason) => {
                eprintln!("{wav}: {reason}");
                continue;
            }
        };
        let audio = resample_to_model_rate(&decoded.samples, 44_100, 16_000).expect("resample");
        let audio: Vec<f32> = audio.into_iter().map(|sample| sample as f32).collect();

        let mut raw = Vec::with_capacity(audio.len() * 4);
        for sample in &audio {
            raw.extend_from_slice(&sample.to_le_bytes());
        }
        std::fs::write(out.join(format!("{index}.f32")), raw).expect("write f32");

        let (pitch, confidence) = estimator.native_outputs(&audio).expect("native output");
        let mut body = String::new();
        let _ = write!(body, "{{\"pitch\":[");
        write_array(&mut body, &pitch);
        let _ = write!(body, "],\"confidence\":[");
        write_array(&mut body, &confidence);
        let _ = write!(body, "]}}");
        std::fs::write(out.join(format!("{index}.json")), body).expect("write json");
        eprintln!("{index}: {} native frames", pitch.len());
    }
}

fn write_array(body: &mut String, values: &[f32]) {
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            body.push(',');
        }
        let _ = write!(body, "{value:.9}");
    }
}
