/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <webgpu/webgpu.h>

#include <cstdint>

namespace executorch::backends::webgpu {

// Caller's instance must enable TimedWaitAny; returns the WaitAny status.
inline WGPUWaitStatus webgpu_wait(WGPUInstance instance, WGPUFuture future) {
  WGPUFutureWaitInfo info = {};
  info.future = future;
  return wgpuInstanceWaitAny(instance, 1, &info, UINT64_MAX);
}

} // namespace executorch::backends::webgpu
