#ifndef KIRA_FRQ_WORLD_SHIM_H_
#define KIRA_FRQ_WORLD_SHIM_H_

#ifdef __cplusplus
extern "C" {
#endif

int kfw_f0_length_dio(int fs, int x_length, double frame_period);

int kfw_f0_length_harvest(int fs, int x_length, double frame_period);

int kfw_dio(const double *x, int x_length, int fs, double f0_floor,
            double f0_ceiling, double frame_period, double *temporal_positions,
            double *f0);

int kfw_harvest(const double *x, int x_length, int fs, double f0_floor,
                double f0_ceiling, double frame_period,
                double *temporal_positions, double *f0);

void kfw_stonemask(const double *x, int x_length, int fs,
                   const double *temporal_positions, const double *f0,
                   int f0_length, double *refined_f0);

// Raw D4C LoveTrain statistic per frame (#55 probe); frames with f0 == 0
// read 0.0. One output value per input f0 frame.
void kfw_d4c_aperiodicity0(const double *x, int x_length, int fs,
                           const double *temporal_positions, const double *f0,
                           int f0_length, double *aperiodicity0);

// Frame-progress hook (#34): installed around one analysis call on the
// calling thread; the vendored loops call it with (stage, done, total).
// Must match kfw_progress_hook_fn in progress_hook.h.
typedef void (*kfw_progress_callback)(void *ctx, int stage, int done,
    int total);
void kfw_install_progress_hook(kfw_progress_callback hook, void *ctx);
void kfw_clear_progress_hook(void);

#ifdef __cplusplus
}
#endif

#endif  // KIRA_FRQ_WORLD_SHIM_H_
