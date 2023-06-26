#include <executorch/core/Runtime.h>
#include <executorch/profiler/profiler.h>

#include <executorch/platform/Platform.h>

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
