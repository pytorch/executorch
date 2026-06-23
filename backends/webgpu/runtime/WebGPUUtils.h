/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <webgpu/webgpu.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace executorch::backends::webgpu::utils {

// Clamp workgroup size to device limit (SwiftShader caps at 128).
inline uint32_t clamp_workgroup_size(WGPUDevice device, uint32_t desired) {
  WGPULimits limits = {};
  if (wgpuDeviceGetLimits(device, &limits) == WGPUStatus_Success &&
      limits.maxComputeInvocationsPerWorkgroup > 0) {
    return std::min(desired, limits.maxComputeInvocationsPerWorkgroup);
  }
  return desired;
}

// 1D dispatch count (mirrors Vulkan div_up); throws if > device limit.
inline uint32_t compute_1d_workgroup_count(
    WGPUDevice device,
    uint32_t num_threads,
    uint32_t workgroup_size,
    const char* op_name) {
  uint32_t count = (num_threads + workgroup_size - 1) / workgroup_size;
  WGPULimits limits = {};
  uint32_t max_count =
      wgpuDeviceGetLimits(device, &limits) == WGPUStatus_Success &&
          limits.maxComputeWorkgroupsPerDimension > 0
      ? limits.maxComputeWorkgroupsPerDimension
      : 65535u; // WebGPU spec-default floor
  if (count > max_count) {
    throw std::runtime_error(
        std::string("WebGPU ") + op_name +
        ": workgroup count exceeds the 1D dispatch limit");
  }
  return count;
}

} // namespace executorch::backends::webgpu::utils
