mod sys;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Estimator {
    Dio,
    Harvest,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct F0Options {
    pub f0_floor_hz: f64,
    pub f0_ceiling_hz: f64,
    pub frame_period_ms: f64,
}

impl Default for F0Options {
    fn default() -> Self {
        Self {
            f0_floor_hz: 71.0,
            f0_ceiling_hz: 800.0,
            frame_period_ms: 5.0,
        }
    }
}
#[derive(Debug, Clone, PartialEq)]
pub struct F0Track {
    pub frame_period_ms: f64,
    pub temporal_positions: Vec<f64>,
    pub f0_hz: Vec<f64>,
}

/// Which analysis pass a frame-progress report belongs to (#34): estimation
/// and StoneMask refinement each walk their own frame grid, so the pipeline
/// composes the two instead of chaining them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgressStage {
    Estimate,
    Refine,
}

/// Optional frame-progress observer (#34): one frames-done/frames-total
/// report per step, each naming its pass total. `Send + Sync` so worker
/// threads can report through it; kept separate from the run reporter so the
/// estimator never learns about wav paths.
pub trait FrameObserver: Send + Sync {
    fn report(&self, stage: ProgressStage, done: usize, total: usize);
}

impl F0Track {
    pub fn len(&self) -> usize {
        self.f0_hz.len()
    }

    pub fn is_empty(&self) -> bool {
        self.f0_hz.is_empty()
    }

    pub fn voiced(&self) -> impl Iterator<Item = f64> + '_ {
        self.f0_hz.iter().copied().filter(|value| *value > 0.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorldError {
    EmptyInput,
    InvalidSampleRate,
    InvalidOptions,
    TooLong,
    AnalysisFailed,
}

impl std::fmt::Display for WorldError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            WorldError::EmptyInput => "input signal is empty",
            WorldError::InvalidSampleRate => "sample rate must be between 1 and 2^31 - 1",
            WorldError::InvalidOptions => {
                "options must have finite values with 0 < floor < ceiling and frame period > 0"
            }
            WorldError::TooLong => "input signal is too long for WORLD's 32-bit sample count",
            WorldError::AnalysisFailed => "WORLD returned an invalid f0 frame count",
        };
        f.write_str(message)
    }
}

impl std::error::Error for WorldError {}

pub fn estimate_f0(
    estimator: Estimator,
    samples: &[f64],
    sample_rate: u32,
    options: &F0Options,
) -> Result<F0Track, WorldError> {
    estimate_f0_with_observer(estimator, samples, sample_rate, options, None)
}

/// [`estimate_f0`] with frame progress: when `observer` is present the
/// vendored loops report through it and the C hook is installed only for the
/// call; `None` means no reporting and no added overhead.
pub fn estimate_f0_with_observer(
    estimator: Estimator,
    samples: &[f64],
    sample_rate: u32,
    options: &F0Options,
    observer: Option<&dyn FrameObserver>,
) -> Result<F0Track, WorldError> {
    validate(samples, sample_rate, options)?;
    sys::analyze(estimator, samples, sample_rate, options, observer)
}

pub fn refine_f0_stonemask(
    samples: &[f64],
    sample_rate: u32,
    track: &mut F0Track,
) -> Result<(), WorldError> {
    refine_f0_stonemask_with_observer(samples, sample_rate, track, None)
}

/// The raw D4C LoveTrain statistic per frame (#55 probe): the ratio D4C
/// compares against its `threshold` (default 0.85) to label a frame unvoiced.
/// High values are periodic (voiced-like), low values aperiodic; frames with
/// `f0 == 0` read `0.0`. One value per track frame, so the caller can sweep a
/// gate threshold without rerunning the estimator.
pub fn d4c_aperiodicity0(
    samples: &[f64],
    sample_rate: u32,
    track: &F0Track,
) -> Result<Vec<f64>, WorldError> {
    if samples.is_empty() {
        return Err(WorldError::EmptyInput);
    }
    if sample_rate == 0 {
        return Err(WorldError::InvalidSampleRate);
    }
    if track.f0_hz.is_empty() {
        return Err(WorldError::AnalysisFailed);
    }
    sys::d4c_aperiodicity0(samples, sample_rate, track)
}

/// [`refine_f0_stonemask`] with frame progress; `None` means no reporting and
/// no added overhead.
pub fn refine_f0_stonemask_with_observer(
    samples: &[f64],
    sample_rate: u32,
    track: &mut F0Track,
    observer: Option<&dyn FrameObserver>,
) -> Result<(), WorldError> {
    if samples.is_empty() {
        return Err(WorldError::EmptyInput);
    }
    if sample_rate == 0 {
        return Err(WorldError::InvalidSampleRate);
    }
    sys::refine_stonemask(samples, sample_rate, track, observer)
}

fn validate(samples: &[f64], sample_rate: u32, options: &F0Options) -> Result<(), WorldError> {
    if samples.is_empty() {
        return Err(WorldError::EmptyInput);
    }
    if sample_rate == 0 {
        return Err(WorldError::InvalidSampleRate);
    }
    if !options.f0_floor_hz.is_finite()
        || !options.f0_ceiling_hz.is_finite()
        || !options.frame_period_ms.is_finite()
        || options.f0_floor_hz <= 0.0
        || options.f0_ceiling_hz <= options.f0_floor_hz
        || options.frame_period_ms <= 0.0
    {
        return Err(WorldError::InvalidOptions);
    }
    Ok(())
}
