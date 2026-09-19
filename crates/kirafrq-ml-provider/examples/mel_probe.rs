//! Local verification probe (not a CI test): dumps the Rust frontend's
//! log-mel and decoded native f0 for a synthetic signal so a Python reference
//! (`librosa` + the real `rmvpe.onnx`) can be diffed against them.
//!
//! Run: `cargo run -p kirafrq-ml-provider --features ort --example mel_probe`

use std::io::Write;

use kirafrq_ml_provider::mel::Frontend;
use kirafrq_ml_provider::policy::UvPolicy;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out = args.get(1).map(String::as_str).unwrap_or("mel_probe.txt");
    let model = args.get(2).cloned();

    // 16 kHz, 3 s: 220 Hz harmonic-rich tone (10 harmonics), matching the
    // scratch bench's `harm220x10`.
    let rate = 16_000.0;
    let len = 3 * 16_000;
    let audio: Vec<f32> = (0..len)
        .map(|index| {
            let t = index as f64 / rate;
            let sum: f64 = (1..=10)
                .map(|k| (std::f64::consts::TAU * 220.0 * k as f64 * t).sin() / k as f64)
                .sum();
            (0.3 * sum / 1.5) as f32
        })
        .collect();

    let mut frontend = Frontend::new();
    let (mel, frames) = frontend.mel(&audio);
    let padded = frontend.pad_to_multiple(&mel, frames);

    let mut file = std::fs::File::create(out).unwrap();
    writeln!(file, "frames {frames}").unwrap();
    writeln!(file, "padded {}", padded.len() / 128).unwrap();
    // The exact input audio, so the reference does not re-synthesize it.
    let audio_path = format!("{out}.f32");
    let mut raw = std::fs::File::create(&audio_path).unwrap();
    for sample in &audio {
        raw.write_all(&sample.to_le_bytes()).unwrap();
    }
    // First mel frame, all 128 bins.
    for mel_bin in 0..128 {
        writeln!(file, "mel {mel_bin} {:.9}", mel[mel_bin * frames]).unwrap();
    }

    if let Some(model) = model {
        // The model path on the 16 kHz signal directly (native grid).
        let estimator =
            kirafrq_ml_provider::rmvpe::Rmvpe::new(model.into(), UvPolicy::rmvpe(None), 1);
        let native = estimator.native_estimate(&audio).expect("native estimate");
        writeln!(file, "native_len {}", native.len()).unwrap();
        for (index, &value) in native.iter().enumerate() {
            writeln!(file, "native {index} {value:.6}").unwrap();
        }

        // And through the public path (44.1 kHz in, table grid out).
        let track = estimator
            .estimate(&vec![0.0; 44_100], None)
            .expect("estimate on the zero buffer");
        writeln!(file, "track_len {}", track.f0_hz.len()).unwrap();
        for (index, &value) in track.f0_hz.iter().enumerate().take(40) {
            writeln!(file, "track {index} {value:.6}").unwrap();
        }
    }
    println!("wrote {out} ({frames} frames)");
}
