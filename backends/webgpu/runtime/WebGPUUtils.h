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
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace executorch::backends::webgpu::utils {

// Ceiling division for non-negative integers (mirrors Vulkan's utils::div_up).
template <typename T>
inline T div_up(T a, T b) {
  return (a + b - 1) / b;
}

// Product of dims (live element count); used by dynamic-resize hooks.
inline uint64_t numel_of(const std::vector<int64_t>& dims) {
  uint64_t n = 1;
  for (int64_t v : dims) {
    if (v < 0) {
      throw std::runtime_error("numel_of: negative dimension");
    }
    n *= static_cast<uint64_t>(v);
  }
  return n;
}

// Clamp workgroup size to device limit (SwiftShader caps at 128).
inline uint32_t clamp_workgroup_size(WGPUDevice device, uint32_t desired) {
  WGPULimits limits = {};
  if (wgpuDeviceGetLimits(device, &limits) == WGPUStatus_Success &&
      limits.maxComputeInvocationsPerWorkgroup > 0) {
    return std::min(desired, limits.maxComputeInvocationsPerWorkgroup);
  }
  return desired;
}

// Clamp to device limit, then floor to pow2 (reduction kernels halve stride).
inline uint32_t clamp_workgroup_size_pow2(WGPUDevice device, uint32_t desired) {
  uint32_t v = clamp_workgroup_size(device, desired);
  uint32_t p = 1u;
  while (p <= (v >> 1u)) {
    p <<= 1u;
  }
  return p;
}

struct WgCount {
  uint32_t x;
  uint32_t y;
};

struct DispatchRange {
  size_t begin;
  size_t end;
};

constexpr bool should_record_q4gsw_dual_route(
    uint32_t max_m,
    bool bicol_eligible,
    bool has_dynamic_shapes) {
  return max_m > 1u && bicol_eligible && has_dynamic_shapes;
}

constexpr bool should_record_sdpa_dual_route(
    bool fd_eligible,
    bool has_dynamic_sequence) {
  return fd_eligible && has_dynamic_sequence;
}

class DispatchRouteRegistry {
 public:
  template <typename IsCompute>
  size_t register_group(
      size_t dispatch_count,
      const std::vector<DispatchRange>& ranges,
      IsCompute&& is_compute) {
    if (dispatch_count < owners_.size() || ranges.size() < 2) {
      throw std::runtime_error("invalid WebGPU dispatch route group");
    }

    std::vector<bool> claimed(dispatch_count, false);
    for (const auto& range : ranges) {
      if (range.begin >= range.end || range.end > dispatch_count) {
        throw std::runtime_error("invalid WebGPU dispatch route range");
      }
      for (size_t i = range.begin; i < range.end; i++) {
        if (!is_compute(i)) {
          throw std::runtime_error(
              "WebGPU dispatch route contains a copy command");
        }
        if (claimed[i] || (i < owners_.size() && owners_[i] != kNoOwner)) {
          throw std::runtime_error("overlapping WebGPU dispatch route ranges");
        }
        claimed[i] = true;
      }
    }

    const size_t group = groups_.size();
    owners_.resize(dispatch_count, kNoOwner);
    for (size_t i = 0; i < claimed.size(); i++) {
      if (claimed[i]) {
        owners_[i] = group;
      }
    }
    groups_.push_back(ranges);
    return group;
  }

  template <typename SetGrid>
  void select(
      size_t group,
      size_t active_route,
      const std::vector<WgCount>& active_grids,
      SetGrid&& set_grid) const {
    if (group >= groups_.size()) {
      throw std::runtime_error("invalid WebGPU dispatch route group");
    }
    const auto& ranges = groups_[group];
    if (active_route >= ranges.size()) {
      throw std::runtime_error("invalid active WebGPU dispatch route");
    }
    const auto& active = ranges[active_route];
    if (active_grids.size() != active.end - active.begin) {
      throw std::runtime_error("WebGPU dispatch route grid count mismatch");
    }
    for (const auto& grid : active_grids) {
      if (grid.x == 0 || grid.y == 0) {
        throw std::runtime_error(
            "active WebGPU dispatch route has a zero grid");
      }
    }

    for (const auto& range : ranges) {
      for (size_t i = range.begin; i < range.end; i++) {
        set_grid(i, {0, 0});
      }
    }
    for (size_t i = 0; i < active_grids.size(); i++) {
      set_grid(active.begin + i, active_grids[i]);
    }
  }

 private:
  static constexpr size_t kNoOwner = static_cast<size_t>(-1);
  std::vector<std::vector<DispatchRange>> groups_;
  std::vector<size_t> owners_;
};

// Device's max workgroups per dispatch dimension; the WebGPU spec-default floor
// (65535) if the query fails — never under-reports a real device's capacity.
inline uint32_t queried_max_workgroups(WGPUDevice device) {
  WGPULimits limits = {};
  return wgpuDeviceGetLimits(device, &limits) == WGPUStatus_Success &&
          limits.maxComputeWorkgroupsPerDimension > 0
      ? limits.maxComputeWorkgroupsPerDimension
      : 65535u;
}

// Pure 2D fold of a 1D workgroup count (device-free, unit-testable): {count,1}
// when count <= max, else a near-square {x, y} with x ~ ceil(sqrt(count)) so
// the launched grid stays close to count. A flat {max, div_up(count, max)}
// split would leave up to ~half the workgroups inactive when count just exceeds
// max, and inactive workgroups still cost launch/scheduling; the near-square
// split keeps the waste to O(sqrt(count)). Throws if even a max*max grid is too
// small (a 3rd dispatch dimension, out of scope). The shader reconstructs the
// linear index from @builtin(num_workgroups), so any x/y factoring works.
inline WgCount fold_workgroup_count_2d(
    uint32_t count,
    uint32_t max_count,
    const char* op_name) {
  if (count <= max_count) {
    return {count, 1u};
  }
  uint32_t x =
      static_cast<uint32_t>(std::ceil(std::sqrt(static_cast<double>(count))));
  x = std::min(x, max_count);
  // ceil-div written overflow-safe (count >= 1 here) as count nears UINT32_MAX.
  uint32_t y = 1u + (count - 1u) / x;
  if (y > max_count) {
    throw std::runtime_error(
        std::string("WebGPU ") + op_name +
        ": workgroup count needs a 3rd dispatch dimension (unsupported)");
  }
  return {x, y};
}

// 1D dispatch count (mirrors Vulkan div_up); throws if > device limit.
inline uint32_t compute_1d_workgroup_count(
    WGPUDevice device,
    uint32_t num_threads,
    uint32_t workgroup_size,
    const char* op_name) {
  uint32_t count = div_up(num_threads, workgroup_size);
  if (count > queried_max_workgroups(device)) {
    throw std::runtime_error(
        std::string("WebGPU ") + op_name +
        ": workgroup count exceeds the 1D dispatch limit");
  }
  return count;
}

// 2D dispatch count: fold the 1D count across x/y when it exceeds the per-dim
// limit (lifts the cap, e.g. for SDPA prefill). Same fast path as compute_1d.
inline WgCount compute_2d_workgroup_count(
    WGPUDevice device,
    uint32_t num_threads,
    uint32_t workgroup_size,
    const char* op_name) {
  uint32_t count = div_up(num_threads, workgroup_size);
  return fold_workgroup_count_2d(
      count, queried_max_workgroups(device), op_name);
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
  return std::min(desired, queried_max_workgroups(device));
}

} // namespace executorch::backends::webgpu::utils
