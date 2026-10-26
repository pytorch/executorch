/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/backends/arm/runtime/VGFExecutionStats.h>

#if defined(EXECUTORCH_VGF_IO_STATS) && EXECUTORCH_VGF_IO_STATS
#include <mutex>

namespace executorch::backends::vgf {
namespace {
thread_local VgfStatsBuffer* capture_buffer = nullptr;
thread_local VgfExecutionStats* execution_stats = nullptr;
std::mutex device_info_mutex;
VgfDeviceInfo device_info;
} // namespace

// cppcheck-suppress unusedFunction
VgfStatsBuffer* set_vgf_stats_buffer(VgfStatsBuffer* buffer) noexcept {
  auto* previous = capture_buffer;
  capture_buffer = buffer;
  return previous;
}
VgfExecutionStats* current_vgf_execution_stats() noexcept {
  return execution_stats;
}
void set_vgf_device_info(const VgfDeviceInfo& info) {
  std::lock_guard<std::mutex> lock(device_info_mutex);
  device_info = info;
}

// cppcheck-suppress unusedFunction
VgfDeviceInfo get_vgf_device_info() {
  std::lock_guard<std::mutex> lock(device_info_mutex);
  return device_info;
}

ScopedVgfExecutionStats::ScopedVgfExecutionStats(const void* handle) noexcept
    : previous_(execution_stats), start_(VgfStatsClock::now()) {
  record_.delegate_handle = reinterpret_cast<uintptr_t>(handle);
  execution_stats = &record_.stats;
}
ScopedVgfExecutionStats::~ScopedVgfExecutionStats() {
  record_.stats.execute_ns = vgf_elapsed_ns(start_);
  execution_stats = previous_;
  if (capture_buffer) {
    if (capture_buffer->size == capture_buffer->capacity) {
      capture_buffer->overflow = true;
    } else {
      capture_buffer->records[capture_buffer->size++] = record_;
    }
  }
}
} // namespace executorch::backends::vgf
#endif
