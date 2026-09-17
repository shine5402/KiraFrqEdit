#include "world_shim.h"

#include <type_traits>

#include "progress_hook.h"
#include "world/dio.h"
#include "world/harvest.h"
#include "world/stonemask.h"

static_assert(std::is_same<kfw_progress_callback, kfw_progress_hook_fn>::value,
    "the shim header and the hook header must agree on the callback type");

thread_local kfw_progress_hook_fn kfw_progress_hook = nullptr;
thread_local void *kfw_progress_hook_ctx = nullptr;

void kfw_install_progress_hook(kfw_progress_callback hook, void *ctx) {
  kfw_progress_hook = hook;
  kfw_progress_hook_ctx = ctx;
}

void kfw_clear_progress_hook(void) {
  kfw_progress_hook = nullptr;
  kfw_progress_hook_ctx = nullptr;
}

int kfw_f0_length_dio(int fs, int x_length, double frame_period) {
  return GetSamplesForDIO(fs, x_length, frame_period);
}

int kfw_f0_length_harvest(int fs, int x_length, double frame_period) {
  return GetSamplesForHarvest(fs, x_length, frame_period);
}

int kfw_dio(const double *x, int x_length, int fs, double f0_floor,
            double f0_ceiling, double frame_period, double *temporal_positions,
            double *f0) {
  DioOption option;
  InitializeDioOption(&option);
  option.f0_floor = f0_floor;
  option.f0_ceil = f0_ceiling;
  option.frame_period = frame_period;
  const int f0_length = GetSamplesForDIO(fs, x_length, frame_period);
  Dio(x, x_length, fs, &option, temporal_positions, f0);
  return f0_length;
}

int kfw_harvest(const double *x, int x_length, int fs, double f0_floor,
                double f0_ceiling, double frame_period,
                double *temporal_positions, double *f0) {
  HarvestOption option;
  InitializeHarvestOption(&option);
  option.f0_floor = f0_floor;
  option.f0_ceil = f0_ceiling;
  option.frame_period = frame_period;
  const int f0_length = GetSamplesForHarvest(fs, x_length, frame_period);
  Harvest(x, x_length, fs, &option, temporal_positions, f0);
  return f0_length;
}

void kfw_stonemask(const double *x, int x_length, int fs,
                   const double *temporal_positions, const double *f0,
                   int f0_length, double *refined_f0) {
  StoneMask(x, x_length, fs, temporal_positions, f0, f0_length, refined_f0);
}
