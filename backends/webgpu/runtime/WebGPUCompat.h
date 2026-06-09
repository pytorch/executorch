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

#if defined(__EMSCRIPTEN__)
#include <emscripten/emscripten.h>
#else
#include <chrono>
#include <thread>
#endif

namespace executorch::backends::webgpu {

// Caller's instance must enable TimedWaitAny; returns the WaitAny status.
inline WGPUWaitStatus webgpu_wait(WGPUInstance instance, WGPUFuture future) {
  WGPUFutureWaitInfo info = {};
  info.future = future;
  return wgpuInstanceWaitAny(instance, 1, &info, UINT64_MAX);
}

// Blocking event-loop pump for the output-map readback (browser: JS yield).
inline void webgpu_poll(WGPUInstance instance) {
#if defined(__EMSCRIPTEN__)
  (void)instance;
  emscripten_sleep(0);
#else
  wgpuInstanceProcessEvents(instance);
  std::this_thread::sleep_for(std::chrono::microseconds(50));
#endif
}

} // namespace executorch::backends::webgpu
