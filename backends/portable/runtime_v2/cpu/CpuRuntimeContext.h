/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/portable/runtime_v2/api/RuntimeContext.h>

namespace executorch {
namespace backends {
namespace portable_v2 {

/**
 * RuntimeContext for the CPU Provider. Currently empty — CPU runs
 * synchronously through the existing portable-kernel registry; no GPU
 * queues, no kernel cache, no per-execute arena. Reserved for future
 * cross-execute caches (e.g., cached layout-conversion buffers).
 *
 * Process-global per-Provider; survives across LoadedDelegate lifetimes.
 */
class CpuRuntimeContext final : public RuntimeContext {};

}  // namespace portable_v2
}  // namespace backends
}  // namespace executorch
