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
#include <cstring>
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

// Create a uniform buffer mapped-at-creation, copy `size` bytes in, and unmap.
inline WGPUBuffer
make_uniform(WGPUDevice device, const void* data, size_t size) {
  WGPUBufferDescriptor desc = {};
  desc.size = size;
  desc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
  desc.mappedAtCreation = true;
  WGPUBuffer buf = wgpuDeviceCreateBuffer(device, &desc);
  if (!buf) {
    throw std::runtime_error("make_uniform: buffer creation failed");
  }
  void* ptr = wgpuBufferGetMappedRange(buf, 0, size);
  if (!ptr) {
    wgpuBufferRelease(buf);
    throw std::runtime_error("make_uniform: mapped range is null");
  }
  std::memcpy(ptr, data, size);
  wgpuBufferUnmap(buf);
  return buf;
}

// Clamp a 1D workgroup count to the device limit, for grid-stride kernels that
// loop over any excess work (vs compute_1d_workgroup_count, which throws).
inline uint32_t clamp_workgroup_count(WGPUDevice device, uint32_t desired) {
  WGPULimits limits = {};
  uint32_t max_count =
      wgpuDeviceGetLimits(device, &limits) == WGPUStatus_Success &&
          limits.maxComputeWorkgroupsPerDimension > 0
      ? limits.maxComputeWorkgroupsPerDimension
      : 65535u; // WebGPU spec-default floor
  return std::min(desired, max_count);
}

} // namespace executorch::backends::webgpu::utils
