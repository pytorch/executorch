#include <executorch/runtime/platform/abort.h>
#include <executorch/runtime/platform/platform.h>

namespace torch {
namespace executor {

/**
 * Trigger the Executorch global runtime to immediately exit without cleaning
 * up, and set an abnormal exit status (platform-defined).
 */
__ET_NORETURN void runtime_abort() {
  et_pal_abort();
}

} // namespace executor
} // namespace torch
