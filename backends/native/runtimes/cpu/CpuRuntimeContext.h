/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/native/core/RuntimeContext.h>

namespace executorch {
namespace backends {
namespace native {

/**
 * RuntimeContext for the CPU Runtime. Currently empty — CPU runs
 * synchronously through the existing portable-kernel registry; no GPU
 * queues, no kernel cache, no per-execute arena. Reserved for future
 * cross-execute caches (e.g., cached layout-conversion buffers).
 *
 * Process-global per-Runtime; survives across DelegateInstance lifetimes.
 */
class CpuRuntimeContext final : public RuntimeContext {};

} // namespace native
} // namespace backends
} // namespace executorch
