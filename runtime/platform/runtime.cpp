/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/platform/profiler.h>
#include <executorch/runtime/platform/runtime.h>

#include <executorch/runtime/platform/platform.h>

namespace executorch {
namespace runtime {

/**
 * Initialize the ExecuTorch global runtime.
 */
void runtime_init() {
  et_pal_init();
  EXECUTORCH_PROFILE_CREATE_BLOCK("default");
}

} // namespace runtime
} // namespace executorch
