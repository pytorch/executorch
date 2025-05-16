/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

namespace executorch::extension::threadpool {

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

} // namespace executorch::extension::threadpool

namespace torch::executorch::threadpool { // DEPRECATED
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces. Note that threadpool incorrectly used
// the namespace `torch::executorch` instead of `torch::executor`.
using ::executorch::extension::threadpool::NoThreadPoolGuard; // DEPRECATED
} // namespace torch::executorch::threadpool
