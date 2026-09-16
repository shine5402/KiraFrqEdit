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

#ifdef __cplusplus
}
#endif

#endif  // KIRA_FRQ_WORLD_SHIM_H_
