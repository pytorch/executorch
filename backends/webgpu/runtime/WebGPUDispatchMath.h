/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// Pure dispatch-grid math with zero WebGPU/Dawn dependency, so it is
// unit-testable without a WGPUDevice (split out of WebGPUUtils.h, which
// requires <webgpu/webgpu.h> for its device-facing functions).

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace executorch::backends::webgpu::utils {

// Ceiling division for non-negative integers (mirrors Vulkan's utils::div_up).
template <typename T>
inline T div_up(T a, T b) {
  return (a + b - 1) / b;
}

// Product of a tensor's dims; the same accumulation was duplicated per-op.
inline uint64_t numel(const std::vector<int64_t>& dims) {
  uint64_t n = 1;
  for (int64_t d : dims) {
    if (d < 0) {
      throw std::runtime_error("numel: negative dimension");
    }
    n *= static_cast<uint64_t>(d);
  }
  return n;
}

// Broadcasts a 1- or 2-element int list to (h, w); PyTorch's convention for
// kernel_size/stride/padding/dilation args. Was duplicated as a local `hw`
// lambda in conv2d/conv_transpose2d/max_pool2d.
inline void parse_hw(
    const std::vector<int64_t>& v,
    uint32_t& h,
    uint32_t& w,
    const char* op_name,
    const char* arg_name) {
  if (v.size() == 1) {
    h = w = static_cast<uint32_t>(v[0]);
  } else if (v.size() == 2) {
    h = static_cast<uint32_t>(v[0]);
    w = static_cast<uint32_t>(v[1]);
  } else {
    throw std::runtime_error(
        std::string("WebGPU ") + op_name + ": " + arg_name +
        " must be 1 or 2 elements");
  }
}

// Adaptive 1D->2D dispatch grid. `count_x`/`count_y` are the dispatch dims;
// `stride_x` (= count_x * wg_size) lets the shader decode a flat index as
// `gid.y * stride_x + gid.x`. Used by ops whose element count can exceed the
// 65535 per-dimension ceiling that compute_1d_workgroup_count throws on.
struct DispatchGrid {
  uint32_t wg_size;
  uint32_t count_x;
  uint32_t count_y;
  uint32_t stride_x;
};

// Given the workgroup count needed (1D) and the device's per-dimension
// dispatch-count ceiling, compute a near-square 2D grid rather than
// {max_dim, div_up(total, max_dim)} — maxing one dim pads the other with
// mostly-idle workgroups (up to ~2x the needed launch) when total isn't a
// clean multiple of max_dim.
inline DispatchGrid compute_dispatch_grid_from_limits(
    uint32_t total, // workgroups needed (1D)
    uint32_t wg_size,
    uint32_t max_dim,
    const char* op_name) {
  DispatchGrid g;
  g.wg_size = wg_size;
  if (total <= max_dim) {
    g.count_x = total;
    g.count_y = 1;
  } else {
    uint32_t sq =
        static_cast<uint32_t>(std::ceil(std::sqrt(static_cast<double>(total))));
    g.count_x = sq < max_dim ? sq : max_dim;
    g.count_y = div_up(total, g.count_x);
    if (g.count_y >
        max_dim) { // > max_dim^2 * wg threads — astronomically large
      throw std::runtime_error(
          std::string("WebGPU ") + op_name +
          ": dispatch exceeds 2D grid capacity");
    }
  }
  g.stride_x = g.count_x * g.wg_size;
  return g;
}

} // namespace executorch::backends::webgpu::utils
