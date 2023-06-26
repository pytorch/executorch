#include <executorch/threadpool/threadpool_guard.h>

namespace torch {
namespace executorch {
namespace threadpool {

thread_local bool NoThreadPoolGuard_enabled = false;

bool NoThreadPoolGuard::is_enabled() {
  return NoThreadPoolGuard_enabled;
}

void NoThreadPoolGuard::set_enabled(bool enabled) {
  NoThreadPoolGuard_enabled = enabled;
}

} // namespace threadpool
} // namespace executorch
} // namespace torch
