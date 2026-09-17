// Frame-progress hook for the vendored WORLD loops (#34).
//
// The shim installs the hook around each analysis call; null means no
// reporting. Instrumented loops invoke kfw_report_progress() as an increment
// plus a call only, so the numerics are untouched. This header lives with the
// shim (outside the vendored tree); the vendored hunks are recorded in
// third_party/World/patches/0001-progress.patch.
#ifndef KIRA_FRQ_WORLD_PROGRESS_HOOK_H_
#define KIRA_FRQ_WORLD_PROGRESS_HOOK_H_

typedef void (*kfw_progress_hook_fn)(void *ctx, int stage, int done, int total);

// Stage ids reported through the hook.
enum {
  KFW_STAGE_ESTIMATE = 0,
  KFW_STAGE_REFINE = 1,
};

extern thread_local kfw_progress_hook_fn kfw_progress_hook;
extern thread_local void *kfw_progress_hook_ctx;

inline void kfw_report_progress(int stage, int done, int total) {
  if (kfw_progress_hook != 0)
    kfw_progress_hook(kfw_progress_hook_ctx, stage, done, total);
}

#endif  // KIRA_FRQ_WORLD_PROGRESS_HOOK_H_
