use std::os::raw::{c_int, c_void};

use crate::{Estimator, F0Options, F0Track, FrameObserver, ProgressStage, WorldError};

/// The C hook installed around one analysis call; `ctx` borrows a
/// [`HookTarget`] for the call's duration. Never unwinds across the boundary.
unsafe extern "C" fn report_trampoline(ctx: *mut c_void, stage: c_int, done: c_int, total: c_int) {
    if ctx.is_null() || done < 0 || total <= 0 {
        return;
    }
    let target = unsafe { &*(ctx as *const HookTarget<'_>) };
    let stage = if stage == 1 {
        ProgressStage::Refine
    } else {
        ProgressStage::Estimate
    };
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        target.observer.report(stage, done as usize, total as usize);
    }));
}

/// Thin carrier for the observer reference: a `&dyn FrameObserver` is fat and
/// cannot cross the C boundary, so the hook gets a pointer to this instead.
struct HookTarget<'a> {
    observer: &'a dyn FrameObserver,
}

unsafe extern "C" {
    fn kfw_f0_length_dio(fs: c_int, x_length: c_int, frame_period: f64) -> c_int;
    fn kfw_f0_length_harvest(fs: c_int, x_length: c_int, frame_period: f64) -> c_int;

    fn kfw_dio(
        x: *const f64,
        x_length: c_int,
        fs: c_int,
        f0_floor: f64,
        f0_ceiling: f64,
        frame_period: f64,
        temporal_positions: *mut f64,
        f0: *mut f64,
    ) -> c_int;

    fn kfw_harvest(
        x: *const f64,
        x_length: c_int,
        fs: c_int,
        f0_floor: f64,
        f0_ceiling: f64,
        frame_period: f64,
        temporal_positions: *mut f64,
        f0: *mut f64,
    ) -> c_int;

    fn kfw_stonemask(
        x: *const f64,
        x_length: c_int,
        fs: c_int,
        temporal_positions: *const f64,
        f0: *const f64,
        f0_length: c_int,
        refined_f0: *mut f64,
    );

    fn kfw_d4c_aperiodicity0(
        x: *const f64,
        x_length: c_int,
        fs: c_int,
        temporal_positions: *const f64,
        f0: *const f64,
        f0_length: c_int,
        aperiodicity0: *mut f64,
    );

    fn kfw_install_progress_hook(
        hook: unsafe extern "C" fn(*mut c_void, c_int, c_int, c_int),
        ctx: *mut c_void,
    );
    fn kfw_clear_progress_hook();
}

fn to_c_int(value: usize, error: WorldError) -> Result<c_int, WorldError> {
    c_int::try_from(value).map_err(|_| error)
}

pub(crate) fn analyze(
    estimator: Estimator,
    samples: &[f64],
    sample_rate: u32,
    options: &F0Options,
    observer: Option<&dyn FrameObserver>,
) -> Result<F0Track, WorldError> {
    let x_length = to_c_int(samples.len(), WorldError::TooLong)?;
    let fs = to_c_int(sample_rate as usize, WorldError::InvalidSampleRate)?;

    let expected = match estimator {
        Estimator::Dio => unsafe { kfw_f0_length_dio(fs, x_length, options.frame_period_ms) },
        Estimator::Harvest => unsafe {
            kfw_f0_length_harvest(fs, x_length, options.frame_period_ms)
        },
    };
    if expected <= 0 {
        return Err(WorldError::AnalysisFailed);
    }

    let mut temporal_positions = vec![0.0f64; expected as usize];
    let mut f0_hz = vec![0.0f64; expected as usize];

    let actual = unsafe {
        // The hook lives on this thread only, for this call only; the
        // analysis runs synchronously between install and clear.
        if let Some(observer) = observer {
            let target = HookTarget { observer };
            kfw_install_progress_hook(report_trampoline, std::ptr::addr_of!(target) as *mut c_void);
        }
        let actual = match estimator {
            Estimator::Dio => kfw_dio(
                samples.as_ptr(),
                x_length,
                fs,
                options.f0_floor_hz,
                options.f0_ceiling_hz,
                options.frame_period_ms,
                temporal_positions.as_mut_ptr(),
                f0_hz.as_mut_ptr(),
            ),
            Estimator::Harvest => kfw_harvest(
                samples.as_ptr(),
                x_length,
                fs,
                options.f0_floor_hz,
                options.f0_ceiling_hz,
                options.frame_period_ms,
                temporal_positions.as_mut_ptr(),
                f0_hz.as_mut_ptr(),
            ),
        };
        kfw_clear_progress_hook();
        actual
    };
    if actual <= 0 || actual as usize > temporal_positions.len() {
        return Err(WorldError::AnalysisFailed);
    }
    temporal_positions.truncate(actual as usize);
    f0_hz.truncate(actual as usize);

    Ok(F0Track {
        frame_period_ms: options.frame_period_ms,
        temporal_positions,
        f0_hz,
    })
}

pub(crate) fn d4c_aperiodicity0(
    samples: &[f64],
    sample_rate: u32,
    temporal_positions: &[f64],
    f0_hz: &[f64],
) -> Result<Vec<f64>, WorldError> {
    let x_length = to_c_int(samples.len(), WorldError::TooLong)?;
    let fs = to_c_int(sample_rate as usize, WorldError::InvalidSampleRate)?;
    let f0_length = to_c_int(f0_hz.len(), WorldError::TooLong)?;
    let mut aperiodicity0 = vec![0.0f64; f0_hz.len()];
    unsafe {
        kfw_d4c_aperiodicity0(
            samples.as_ptr(),
            x_length,
            fs,
            temporal_positions.as_ptr(),
            f0_hz.as_ptr(),
            f0_length,
            aperiodicity0.as_mut_ptr(),
        );
    }
    Ok(aperiodicity0)
}

pub(crate) fn refine_stonemask(
    samples: &[f64],
    sample_rate: u32,
    track: &mut F0Track,
    observer: Option<&dyn FrameObserver>,
) -> Result<(), WorldError> {
    let x_length = to_c_int(samples.len(), WorldError::TooLong)?;
    let fs = to_c_int(sample_rate as usize, WorldError::InvalidSampleRate)?;
    let f0_length = to_c_int(track.f0_hz.len(), WorldError::TooLong)?;
    if f0_length == 0 {
        return Err(WorldError::AnalysisFailed);
    }

    let mut refined_f0 = vec![0.0f64; track.f0_hz.len()];
    unsafe {
        if let Some(observer) = observer {
            let target = HookTarget { observer };
            kfw_install_progress_hook(report_trampoline, std::ptr::addr_of!(target) as *mut c_void);
        }
        kfw_stonemask(
            samples.as_ptr(),
            x_length,
            fs,
            track.temporal_positions.as_ptr(),
            track.f0_hz.as_ptr(),
            f0_length,
            refined_f0.as_mut_ptr(),
        );
        kfw_clear_progress_hook();
    }
    track.f0_hz = refined_f0;
    Ok(())
}
