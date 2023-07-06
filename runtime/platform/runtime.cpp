#include <executorch/runtime/platform/profiler.h>
#include <executorch/runtime/platform/runtime.h>

#include <executorch/runtime/platform/platform.h>

namespace torch {
namespace executor {

/**
 * Initialize the Executorch global runtime.
 */
void runtime_init() {
  et_pal_init();
  EXECUTORCH_PROFILE_CREATE_BLOCK("default");
}

} // namespace executor
} // namespace torch
