/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <webgpu/webgpu.h>

#if defined(__EMSCRIPTEN__)
#include <emscripten/emscripten.h>
#else
#include <chrono>
#include <thread>
#endif

namespace executorch::backends::webgpu {

// Drive pending callbacks; callers loop until their done flag is set. The
// native (Dawn) build pumps the instance event queue then briefly yields the
// CPU so the caller's wait loop does not busy-spin a core at 100%
// (wgpuInstanceProcessEvents is non-blocking); the browser yields to the JS
// event loop.
inline void webgpu_poll(WGPUInstance instance, WGPUDevice device) {
#if defined(__EMSCRIPTEN__)
  (void)instance;
  (void)device;
  emscripten_sleep(0);
#else
  (void)device;
  wgpuInstanceProcessEvents(instance);
  std::this_thread::sleep_for(std::chrono::microseconds(50));
#endif
}

} // namespace executorch::backends::webgpu
