/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/platform/abort.h>
#include <executorch/runtime/platform/platform.h>

namespace executorch {
namespace runtime {

/**
 * Trigger the ExecuTorch global runtime to immediately exit without cleaning
 * up, and set an abnormal exit status (platform-defined).
 */
__ET_NORETURN void runtime_abort() {
  et_pal_abort();
}

} // namespace runtime
} // namespace executorch
