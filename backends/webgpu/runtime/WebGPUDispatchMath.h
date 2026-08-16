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
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace executorch::backends::webgpu::utils {

// Ceiling division for non-negative integers (mirrors Vulkan's utils::div_up).
template <typename T>
inline T div_up(T a, T b) {
  return a / b + (a % b != 0);
}

// Product of a tensor's dims; the same accumulation was duplicated per-op.
inline uint64_t numel(const std::vector<int64_t>& dims) {
  uint64_t n = 1;
  for (int64_t d : dims) {
    if (d < 0) {
      throw std::runtime_error("numel: negative dimension");
    }
    uint64_t ud = static_cast<uint64_t>(d);
    if (ud != 0 && n > UINT64_MAX / ud) {
      throw std::runtime_error("numel: element count overflow");
    }
    n *= ud;
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
    bool has_dynamic_shapes,
    bool record_q4gsw_decode_route) {
  return max_m > 1u && bicol_eligible &&
      (has_dynamic_shapes || record_q4gsw_decode_route);
}

constexpr bool should_record_sdpa_dual_route(
    bool fd_eligible,
    bool has_dynamic_sequence,
    bool has_dynamic_position) {
  return fd_eligible && (has_dynamic_sequence || has_dynamic_position);
}

constexpr bool is_q4gsw_bk64_eligible(
    uint32_t k,
    uint32_t n,
    uint32_t group_size,
    bool has_bias,
    bool shader_f16_supported,
    uint32_t max_invocations,
    uint32_t max_workgroup_storage_bytes) {
  constexpr uint32_t kRequiredInvocations = 256u;
  constexpr uint32_t kRequiredStorageBytes = 2u * 64u * 64u * sizeof(uint16_t);
  const bool ordinary_llama_projection = (k == 2048u && n == 8192u) ||
      (k == 8192u && n == 2048u) || (k == 2048u && n == 2048u);
  return ordinary_llama_projection && k % 64u == 0u && group_size == 64u &&
      !has_bias && shader_f16_supported &&
      max_invocations >= kRequiredInvocations &&
      max_workgroup_storage_bytes >= kRequiredStorageBytes;
}

constexpr bool is_q4gsw_bk64_live_m(uint32_t m) {
  return m == 128u || m == 508u || m == 512u;
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
