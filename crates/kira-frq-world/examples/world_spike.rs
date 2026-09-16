use std::error::Error;
use std::fs::{self, File};
use std::io::BufWriter;
use std::path::Path;
use std::time::Instant;

use kira_frq_world::{Estimator, F0Options, F0Track, estimate_f0, refine_f0_stonemask};

const SAMPLE_RATE: u32 = 44_100;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut options = F0Options::default();
    let mut seconds = 4.0;
    let mut wav_arg: Option<String> = None;
    let mut index = 0;
    while index < args.len() {
        let value = args
            .get(index + 1)
            .ok_or_else(|| format!("{} needs a value", args[index]))?
            .clone();
        match args[index].as_str() {
            "--floor" => options.f0_floor_hz = value.parse()?,
            "--ceiling" => options.f0_ceiling_hz = value.parse()?,
            "--seconds" => seconds = value.parse()?,
            "--wav" => wav_arg = Some(value),
            other => return Err(format!("unknown argument: {other}").into()),
        }
        index += 2;
    }

    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.local/world-spike");
    fs::create_dir_all(&out_dir)?;

    let (samples, sample_rate, input_label) = match &wav_arg {
        Some(path) => {
            let (samples, sample_rate) = read_wav(Path::new(path))?;
            (samples, sample_rate, path.clone())
        }
        None => {
            let samples = sweep(100.0, 500.0, seconds, SAMPLE_RATE);
            let path = out_dir.join("sweep_100_500Hz.wav");
            write_wav(&path, &samples, SAMPLE_RATE)?;
            (samples, SAMPLE_RATE, path.display().to_string())
        }
    };
    println!(
        "input: {input_label} ({:.1} s, {sample_rate} Hz, {} samples)",
        samples.len() as f64 / sample_rate as f64,
        samples.len(),
    );
    println!(
        "options: floor {} Hz, ceiling {} Hz, frame period {} ms",
        options.f0_floor_hz, options.f0_ceiling_hz, options.frame_period_ms
    );

    for estimator in [Estimator::Dio, Estimator::Harvest] {
        let name = match estimator {
            Estimator::Dio => "dio",
            Estimator::Harvest => "harvest",
        };

        let start = Instant::now();
        let raw = estimate_f0(estimator, &samples, sample_rate, &options)?;
        let raw_time = start.elapsed();
        report(name, &raw, samples.len(), sample_rate, raw_time);

        let mut refined = raw;
        let start = Instant::now();
        refine_f0_stonemask(&samples, sample_rate, &mut refined)?;
        let voiced = refined.voiced().count();
        println!(
            "  +StoneMask: {:.1} ms, voiced {voiced}, f0 mean {:.2} Hz",
            start.elapsed().as_secs_f64() * 1e3,
            refined.voiced().sum::<f64>() / voiced as f64
        );

        if wav_arg.is_none() {
            let csv = out_dir.join(format!("sweep_{name}_f0.csv"));
            write_f0_csv(&csv, &refined)?;
            println!("  f0 CSV: {}", csv.display());
        }
    }

    Ok(())
}

fn read_wav(path: &Path) -> Result<(Vec<f64>, u32), Box<dyn Error>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let channels = spec.channels as usize;
    let samples: Vec<f64> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .map(|sample| sample.map(f64::from))
            .collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1i64 << (spec.bits_per_sample - 1)) as f64;
            reader
                .samples::<i32>()
                .map(|sample| sample.map(|value| value as f64 * scale))
                .collect::<Result<_, _>>()?
        }
    };
    let mono = if channels == 1 {
        samples
    } else {
        samples
            .chunks(channels)
            .map(|frame| frame.iter().sum::<f64>() / channels as f64)
            .collect()
    };
    Ok((mono, spec.sample_rate))
}

fn sweep(start_hz: f64, end_hz: f64, seconds: f64, sample_rate: u32) -> Vec<f64> {
    let count = (sample_rate as f64 * seconds) as usize;
    let rate = (end_hz - start_hz) / seconds;
    (0..count)
        .map(|n| {
            let t = n as f64 / sample_rate as f64;
            (1..=8)
                .map(|harmonic| {
                    let phase = 2.0
                        * std::f64::consts::PI
                        * (start_hz * harmonic as f64 * t + 0.5 * rate * harmonic as f64 * t * t);
                    (0.5 / harmonic as f64) * phase.sin()
                })
                .sum()
        })
        .collect()
}

fn write_wav(path: &Path, samples: &[f64], sample_rate: u32) -> Result<(), Box<dyn Error>> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::new(BufWriter::new(File::create(path)?), spec)?;
    for sample in samples {
        writer.write_sample((sample.clamp(-1.0, 1.0) * i16::MAX as f64) as i16)?;
    }
    writer.finalize()?;
    Ok(())
}

fn write_f0_csv(path: &Path, track: &F0Track) -> Result<(), Box<dyn Error>> {
    let mut csv = String::from("time_ms,f0_hz\n");
    for (position, f0) in track.temporal_positions.iter().zip(&track.f0_hz) {
        csv.push_str(&format!("{:.3},{f0:.3}\n", position * 1e3));
    }
    fs::write(path, csv)?;
    Ok(())
}

fn report(
    name: &str,
    track: &F0Track,
    sample_count: usize,
    sample_rate: u32,
    elapsed: std::time::Duration,
) {
    let expected = (sample_count as f64 / (sample_rate as f64 * track.frame_period_ms / 1000.0))
        .floor() as usize
        + 1;
    let mut voiced: Vec<f64> = track.voiced().collect();
    voiced.sort_by(f64::total_cmp);
    let mean = voiced.iter().sum::<f64>() / voiced.len() as f64;
    println!(
        "\n{name}: {:.1} ms ({:.0}x realtime), {} frames (expected {expected}), grid {:.1} ms",
        elapsed.as_secs_f64() * 1e3,
        (sample_count as f64 / sample_rate as f64) / elapsed.as_secs_f64(),
        track.len(),
        track.frame_period_ms
    );
    println!(
        "  voiced {}/{}, f0 min {:.2} Hz, mean {:.2} Hz, max {:.2} Hz",
        voiced.len(),
        track.len(),
        voiced.first().copied().unwrap_or(0.0),
        mean,
        voiced.last().copied().unwrap_or(0.0)
    );
}
