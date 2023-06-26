#pragma once

namespace torch {
namespace executorch {
namespace threadpool {

// A RAII, thread local (!) guard that enables or disables guard upon
// construction, and sets it back to the original value upon destruction.
struct NoThreadPoolGuard {
  static bool is_enabled();
  static void set_enabled(bool enabled);

  NoThreadPoolGuard() : prev_mode_(NoThreadPoolGuard::is_enabled()) {
    NoThreadPoolGuard::set_enabled(true);
  }
  ~NoThreadPoolGuard() {
    NoThreadPoolGuard::set_enabled(prev_mode_);
  }

 private:
  const bool prev_mode_;
};

} // namespace threadpool
} // namespace executorch
} // namespace torch
