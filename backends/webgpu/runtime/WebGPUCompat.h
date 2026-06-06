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

// Make progress on pending WebGPU callbacks; callers loop until their done flag
// is set. Native (Dawn): pump the event queue + brief yield (no busy-spin).
// Browser: yield to the JS event loop.
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
