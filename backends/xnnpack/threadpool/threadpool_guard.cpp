/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/xnnpack/threadpool/threadpool_guard.h>

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
