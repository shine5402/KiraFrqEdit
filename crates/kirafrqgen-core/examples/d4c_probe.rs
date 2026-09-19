//! Throwaway probe for wayfinder #55: dump Harvest's f0 (post-StoneMask), the
//! raw D4C LoveTrain statistic, and the table-grid amplitude per frame to CSV,
//! so a D4C voicing-gate threshold can be swept offline against the RMVPE
//! reference. NOT a product surface — captured on the #55 prototype branch.
//!
//! Usage: d4c_probe <wav-or-dir> <out-dir>

use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use kirafrq_audio::{DecodeOutcome, decode_wav};
use kirafrq_world_binding::{
    Estimator, F0Options, d4c_aperiodicity0, estimate_f0, refine_f0_stonemask,
};
use rayon::prelude::*;

const SAMPLE_RATE: u32 = 44_100;
const HOP: usize = 256;

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let input = PathBuf::from(
        args.next()
            .ok_or("usage: d4c_probe <wav-or-dir> <out-dir>")?,
    );
    let out_dir = PathBuf::from(
        args.next()
            .ok_or("usage: d4c_probe <wav-or-dir> <out-dir>")?,
    );
    fs::create_dir_all(&out_dir)?;

    let wavs: Vec<PathBuf> = if input.is_dir() {
        let mut found: Vec<PathBuf> = fs::read_dir(&input)?
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| {
                path.extension()
                    .is_some_and(|extension| extension.eq_ignore_ascii_case("wav"))
            })
            .collect();
        found.sort();
        found
    } else {
        vec![input.clone()]
    };

    let results: Vec<Result<PathBuf, String>> = wavs
        .par_iter()
        .map(|wav| process(wav, &out_dir).map_err(|error| format!("{}: {error}", wav.display())))
        .collect();

    let mut ok = 0usize;
    for result in &results {
        match result {
            Ok(path) => {
                ok += 1;
                println!("{}", path.display());
            }
            Err(message) => eprintln!("{message}"),
        }
    }
    println!("{ok}/{} wavs", results.len());
    Ok(())
}

fn process(wav: &Path, out_dir: &Path) -> Result<PathBuf, Box<dyn Error>> {
    let samples = match decode_wav(wav) {
        DecodeOutcome::Decoded(decoded) => decoded.samples,
        DecodeOutcome::Empty => return Err("empty wav".into()),
        DecodeOutcome::Failed(reason) => return Err(reason.into()),
    };

    let options = F0Options {
        frame_period_ms: HOP as f64 / SAMPLE_RATE as f64 * 1000.0,
        ..F0Options::default()
    };
    let start = std::time::Instant::now();
    let mut track = estimate_f0(Estimator::Harvest, &samples, SAMPLE_RATE, &options)?;
    refine_f0_stonemask(&samples, SAMPLE_RATE, &mut track)?;
    let estimate_ms = start.elapsed().as_secs_f64() * 1e3;
    let start = std::time::Instant::now();
    let d4c0 = d4c_aperiodicity0(&samples, SAMPLE_RATE, &track)?;
    let d4c_ms = start.elapsed().as_secs_f64() * 1e3;
    eprintln!(
        "{}: {:.0} ms harvest+stonemask, {:.0} ms d4c ({:.1}% of estimate), {} frames",
        wav.display(),
        estimate_ms,
        d4c_ms,
        100.0 * d4c_ms / estimate_ms.max(1e-9),
        track.len()
    );

    let mut csv = String::from("frame,t_ms,f0_hz,d4c0,amp\n");
    for (index, ((t, f0), aperiodicity)) in track
        .temporal_positions
        .iter()
        .zip(&track.f0_hz)
        .zip(&d4c0)
        .enumerate()
    {
        csv.push_str(&format!(
            "{index},{:.4},{f0:.4},{aperiodicity:.6},{:.4}\n",
            t * 1e3,
            frame_amplitude(&samples, index)
        ));
    }

    let stem = wav
        .file_stem()
        .ok_or("unnamed wav")?
        .to_string_lossy()
        .into_owned();
    let path = out_dir.join(format!("{stem}.csv"));
    fs::write(&path, csv)?;
    Ok(path)
}

/// `2^15 * mean(|x|)` over frame `i`'s window, the table/energy-gate amplitude
/// (mirrors `kirafrqgen-core::table::amplitudes`).
fn frame_amplitude(samples: &[f64], frame: usize) -> f64 {
    let lo = frame * HOP;
    let hi = (lo + HOP).min(samples.len());
    if hi <= lo {
        return 0.0;
    }
    let sum: f64 = samples[lo..hi].iter().map(|sample| sample.abs()).sum();
    32_768.0 / (hi - lo) as f64 * sum
}
