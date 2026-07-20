/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <webgpu/webgpu.h>

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace executorch::backends::webgpu {

#ifdef WGPU_BACKEND_ENABLE_PROFILING

// Per-dispatch GPU timing; mirrors Vulkan QueryPool ShaderDuration.
struct ShaderDuration {
  uint32_t idx = 0;
  std::string kernel_name;
  std::array<uint32_t, 3> global_wg{};
  std::array<uint32_t, 3> local_wg{};
  uint64_t start_time_ns = 0;
  uint64_t end_time_ns = 0;
  uint64_t execution_duration_ns = 0;
};

// GPU timestamp-query pool; re-port of Vulkan vk_api/QueryPool.
class WebGPUQueryPool {
 public:
  WebGPUQueryPool() = default;
  ~WebGPUQueryPool();

  WebGPUQueryPool(const WebGPUQueryPool&) = delete;
  WebGPUQueryPool& operator=(const WebGPUQueryPool&) = delete;

  // Create the QuerySet + readback buffers; query the ns-per-tick period.
  void initialize(WGPUDevice device, uint32_t max_pairs);
  bool is_initialized() const {
    return qset_ != nullptr;
  }
  uint32_t capacity() const {
    return capacity_pairs_;
  }

  // Clear durations and set the dispatch count for this run.
  void reset(uint32_t num_dispatches);

  // timestampWrites for pass i: begin=2i, end=2i+1.
  WGPUPassTimestampWrites writes_for(uint32_t i);

  // Record pass i's label + workgroup sizes (start/end filled by extract).
  void record(
      uint32_t i,
      const std::string& name,
      std::array<uint32_t, 3> gwg,
      std::array<uint32_t, 3> lwg);

  // Resolve the QuerySet into the readback buffer; call before submit.
  void resolve(WGPUCommandEncoder encoder);

  // Map the readback, convert ticks->ns, fill durations; call after submit.
  void extract_results(WGPUInstance instance);

  const std::vector<ShaderDuration>& results() const {
    return durations_;
  }
  void print_results(bool tsv = false) const;
  uint64_t get_mean_shader_ns(const std::string& kernel_name) const;

 private:
  WGPUQuerySet qset_ = nullptr;
  WGPUBuffer resolve_buf_ = nullptr; // QueryResolve | CopySrc
  WGPUBuffer readback_buf_ = nullptr; // MapRead | CopyDst
  uint32_t capacity_pairs_ = 0;
  uint32_t num_pairs_ = 0;
  double ns_per_tick_ = 1.0; // WebGPU timestamps are already nanoseconds
  std::vector<ShaderDuration> durations_;
};

#endif // WGPU_BACKEND_ENABLE_PROFILING

} // namespace executorch::backends::webgpu
