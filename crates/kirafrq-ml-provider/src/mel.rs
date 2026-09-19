//! The RMVPE log-mel frontend: periodic Hann window, magnitude spectrum (not
//! squared) projected through an HTK mel filterbank with Slaney
//! normalization, `log(max(mel, 1e-5))`, frames at 16 kHz with hop 160 and a
//! **centered** reflect-padded STFT — RVC's `infer/rmvpe.py` settings
//! (`torch.stft(center=True)`, `pad_mode="reflect"`), which `rmvpe_onnx`
//! reproduces with `np.pad(mode="reflect")`. Verified against both locally.
//!
//! Exposed (as a hidden API) so the local verification probe can dump its
//! values; the pipeline never calls it directly.

use std::sync::Arc;

use realfft::RealFftPlanner;
use realfft::num_complex::Complex32;

/// The model's frame rate, from the RMVPE contract.
pub const SAMPLE_RATE: u32 = 16_000;
/// STFT size, `n_fft = 1024` in RVC's frontend.
pub const N_FFT: usize = 1024;
/// Hop between frames, 160 samples (10 ms at [`SAMPLE_RATE`]).
pub const HOP: usize = 160;
/// Mel bands the RMVPE graph expects.
pub const N_MELS: usize = 128;
/// Lowest mel filter edge, Hz.
pub const FMIN_HZ: f64 = 30.0;
/// Highest mel filter edge, Hz.
pub const FMAX_HZ: f64 = 8000.0;
/// Lower clamp before the log, matching RVC's `clamp=1e-5`.
pub const MEL_CLAMP: f32 = 1e-5;

/// Zero-pad the mel to the next multiple of 32 frames (at least 32), the
/// shape RMVPE's graph requires.
pub const PAD_MULTIPLE: usize = 32;

/// The log-mel frontend's scratch space, reusable across files.
pub struct Frontend {
    fft: Arc<dyn realfft::RealToComplex<f32>>,
    window: Vec<f32>,
    /// `[N_MELS][N_FFT/2+1]`, row-major.
    basis: Vec<f32>,
    scratch: Vec<Complex32>,
    spectrum: Vec<Complex32>,
    frame: Vec<f32>,
}

impl Frontend {
    /// The frontend with its FFT plan, window and filterbank ready.
    pub fn new() -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(N_FFT);
        let scratch = vec![Complex32::default(); fft.get_scratch_len()];
        Self {
            window: hann(N_FFT),
            basis: mel_basis(),
            scratch,
            spectrum: vec![Complex32::default(); N_FFT / 2 + 1],
            frame: vec![0.0; N_FFT],
            fft,
        }
    }

    /// The log-mel of `audio_16k`: `[N_MELS * frames]` row-major (`m`-major,
    /// the model's `[1, 128, T]` layout) plus the unpadded frame count.
    /// Frames are centered: frame `k` covers samples
    /// `[k*HOP - N_FFT/2, k*HOP + N_FFT/2)`, reflect-padded at both ends, so
    /// there are `1 + L/HOP` frames and native frame `k` sits at
    /// `t = k*HOP/16000` (#47's origin 0).
    pub fn mel(&mut self, audio_16k: &[f32]) -> (Vec<f32>, usize) {
        let frames = audio_16k.len() / HOP + 1;
        let half = N_FFT / 2;
        let mut mel = vec![0.0_f32; N_MELS * frames];
        for frame in 0..frames {
            let center = frame * HOP;
            for (index, slot) in self.frame.iter_mut().enumerate() {
                // Sample index `center - half + index`, reflect-padded.
                let position = center as isize + index as isize - half as isize;
                *slot = reflect(audio_16k, position) * self.window[index];
            }
            self.fft
                .process_with_scratch(&mut self.frame, &mut self.spectrum, &mut self.scratch)
                .expect("the FFT plan matches the buffers");
            let n_freqs = N_FFT / 2 + 1;
            for mel_bin in 0..N_MELS {
                let row = &self.basis[mel_bin * n_freqs..(mel_bin + 1) * n_freqs];
                let mut energy = 0.0_f32;
                for (bin, &weight) in row.iter().enumerate() {
                    let spectrum = self.spectrum[bin];
                    let magnitude = (spectrum.re * spectrum.re + spectrum.im * spectrum.im).sqrt();
                    energy += weight * magnitude;
                }
                mel[mel_bin * frames + frame] = energy.max(MEL_CLAMP).ln();
            }
        }
        (mel, frames)
    }

    /// Zero-pad the mel to `[N_MELS * T_pad]` with `T_pad` the next multiple
    /// of [`PAD_MULTIPLE`].
    pub fn pad_to_multiple(&self, mel: &[f32], frames: usize) -> Vec<f32> {
        let padded = frames.max(1).div_ceil(PAD_MULTIPLE) * PAD_MULTIPLE;
        let mut out = vec![0.0_f32; N_MELS * padded];
        for mel_bin in 0..N_MELS {
            let source = &mel[mel_bin * frames..(mel_bin + 1) * frames];
            out[mel_bin * padded..mel_bin * padded + frames].copy_from_slice(source);
        }
        out
    }
}

impl Default for Frontend {
    fn default() -> Self {
        Self::new()
    }
}

/// Periodic Hann window (`sym = false`), matching `torch.hann_window`'s
/// default in RVC's frontend.
fn hann(n: usize) -> Vec<f32> {
    (0..n)
        .map(|index| 0.5 - 0.5 * (std::f32::consts::TAU * index as f32 / n as f32).cos())
        .collect()
}

/// Reflect padding (`np.pad(..., mode="reflect")` / torch's `reflect`):
/// index `-1` reads sample 1, `len` reads `len - 2`. A one-sample signal has
/// nothing to reflect and reads itself.
fn reflect(samples: &[f32], position: isize) -> f32 {
    let len = samples.len() as isize;
    if len <= 1 {
        return samples.first().copied().unwrap_or(0.0);
    }
    let mut index = position;
    loop {
        if index < 0 {
            index = -index;
        } else if index >= len {
            index = 2 * len - 2 - index;
        } else {
            return samples[index as usize];
        }
    }
}

fn hz_to_mel_htk(hz: f64) -> f64 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz_htk(mel: f64) -> f64 {
    700.0 * (10_f64.powf(mel / 2595.0) - 1.0)
}

/// The mel filterbank `librosa.filters.mel(sr=16000, n_fft=1024, n_mels=128,
/// fmin=30, fmax=8000, htk=True)` builds for RVC's frontend: triangular
/// filters with a peak of 1, *including* librosa's Slaney area normalization
/// (the `norm="slaney"` default; `htk=True` only switches the scale).
fn mel_basis() -> Vec<f32> {
    let n_freqs = N_FFT / 2 + 1;
    let mel_min = hz_to_mel_htk(FMIN_HZ);
    let mel_max = hz_to_mel_htk(FMAX_HZ);
    let edges: Vec<f64> = (0..N_MELS + 2)
        .map(|index| {
            let mel = mel_min + (mel_max - mel_min) * index as f64 / (N_MELS + 1) as f64;
            mel_to_hz_htk(mel)
        })
        .collect();
    let bin_hz: Vec<f64> = (0..n_freqs)
        .map(|bin| bin as f64 * SAMPLE_RATE as f64 / N_FFT as f64)
        .collect();

    let mut basis = vec![0.0_f32; N_MELS * n_freqs];
    for mel_bin in 0..N_MELS {
        let (lo, mid, hi) = (edges[mel_bin], edges[mel_bin + 1], edges[mel_bin + 2]);
        // Slaney normalization: each filter has approximately unit energy.
        let enorm = 2.0 / (edges[mel_bin + 2] - edges[mel_bin]);
        for bin in 0..n_freqs {
            let hz = bin_hz[bin];
            let weight = if hz <= lo || hz >= hi {
                0.0
            } else if hz <= mid {
                (hz - lo) / (mid - lo)
            } else {
                (hi - hz) / (hi - mid)
            };
            basis[mel_bin * n_freqs + bin] = (weight * enorm) as f32;
        }
    }
    basis
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_hann_window_is_periodic() {
        let window = hann(8);
        assert_eq!(window[0], 0.0);
        // A symmetric window would end at 0.0 too; the periodic one does not.
        assert!(window[7] > 0.1, "{:?}", window);
        assert!((window[4] - 1.0).abs() < 1e-6, "{:?}", window);
    }

    #[test]
    fn the_mel_basis_has_unit_peaks_and_no_empty_bands() {
        let basis = mel_basis();
        assert_eq!(basis.len(), N_MELS * (N_FFT / 2 + 1));
        for mel_bin in 0..N_MELS {
            let row = &basis[mel_bin * (N_FFT / 2 + 1)..(mel_bin + 1) * (N_FFT / 2 + 1)];
            let peak = row.iter().copied().fold(f32::MIN, f32::max);
            assert!(peak > 0.0, "band {mel_bin} is empty");
            assert!(peak <= 1.0 + 1e-6, "band {mel_bin} peak {peak}");
        }
    }

    #[test]
    fn a_low_band_sits_below_a_high_band() {
        // The 440 Hz bin (bin 28) must fall in a lower mel band than the
        // 4 kHz bin (bin 256).
        let basis = mel_basis();
        let n_freqs = N_FFT / 2 + 1;
        let band_of = |bin: usize| {
            (0..N_MELS)
                .find(|&mel_bin| basis[mel_bin * n_freqs + bin] > 0.0)
                .expect("bin in some band")
        };
        assert!(band_of(28) < band_of(256));
    }

    #[test]
    fn frames_are_centered_with_a_reflect_pad() {
        let mut frontend = Frontend::new();
        // 1600 samples: frames at 0, 160, ..., 1600 -> 11 frames.
        let (mel, frames) = frontend.mel(&vec![0.0; 1600]);
        assert_eq!(frames, 1600 / HOP + 1);
        assert_eq!(frames, 11);
        assert_eq!(mel.len(), N_MELS * frames);

        // A 1-sample input still yields one frame.
        let (_, one) = frontend.mel(&[1.0]);
        assert_eq!(one, 1);
    }

    #[test]
    fn reflect_padding_mirrors_around_the_edges() {
        let samples = [0.0_f32, 1.0, 2.0, 3.0];
        assert_eq!(reflect(&samples, 0), 0.0);
        assert_eq!(reflect(&samples, 3), 3.0);
        assert_eq!(reflect(&samples, -1), 1.0);
        assert_eq!(reflect(&samples, -3), 3.0);
        assert_eq!(reflect(&samples, 4), 2.0);
        assert_eq!(reflect(&samples, 6), 0.0);
        // A one-sample signal has nothing to reflect.
        assert_eq!(reflect(&[0.5], -5), 0.5);
        assert_eq!(reflect(&[0.5], 7), 0.5);
    }

    #[test]
    fn the_frontend_pads_up_to_the_next_multiple_of_32() {
        let frontend = Frontend::new();
        for (frames, padded) in [(1, 32), (31, 32), (32, 32), (33, 64), (64, 64)] {
            let mel = vec![0.0_f32; N_MELS * frames];
            assert_eq!(
                frontend.pad_to_multiple(&mel, frames).len(),
                N_MELS * padded
            );
        }
    }
}
