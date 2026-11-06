/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/api/ExternalIO.h>

#ifdef VK_ANDROID_external_memory_android_hardware_buffer

#include <map>
#include <mutex>
#include <utility>

namespace vkcompute {
namespace api {

namespace {

struct Registry {
  std::mutex mutex;
  std::map<std::pair<bool, size_t>, AHardwareBuffer*> buffers;
};

Registry& registry() {
  static Registry r;
  return r;
}

} // namespace

void set_external_io_buffer(
    const bool is_input,
    const size_t index,
    AHardwareBuffer* ahb) {
  std::lock_guard<std::mutex> lock(registry().mutex);
  if (ahb == nullptr) {
    registry().buffers.erase({is_input, index});
  } else {
    registry().buffers[{is_input, index}] = ahb;
  }
}

std::map<std::pair<bool, size_t>, AHardwareBuffer*> take_external_io_buffers() {
  std::lock_guard<std::mutex> lock(registry().mutex);
  return std::exchange(registry().buffers, {});
}

void clear_external_io_buffers() {
  std::lock_guard<std::mutex> lock(registry().mutex);
  registry().buffers.clear();
}

} // namespace api
} // namespace vkcompute

#endif /* VK_ANDROID_external_memory_android_hardware_buffer */
