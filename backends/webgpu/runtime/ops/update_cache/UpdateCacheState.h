/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/webgpu/runtime/WebGPUDispatchMath.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace executorch::backends::webgpu {

struct UpdateCacheParams {
  uint32_t numel;
  uint32_t dst_offset;
  uint32_t cache_numel;
  uint32_t _pad0;
};
static_assert(
    sizeof(UpdateCacheParams) == 16,
    "UpdateCacheParams must be 16 bytes");

struct LiveUpdateCacheInputs {
  std::vector<int64_t> value_dims;
  std::vector<int64_t> cache_dims;
  size_t expected_value_rank;
  size_t expected_cache_rank;
  int64_t start_pos;
  uint32_t workgroup_size;
  uint32_t max_workgroups_per_dimension;
};

struct LiveUpdateCacheState {
  UpdateCacheParams params;
  uint32_t workgroup_count_x;
};

inline LiveUpdateCacheState compute_live_update_cache_state(
    const LiveUpdateCacheInputs& inputs) {
  if (inputs.value_dims.size() != inputs.expected_value_rank ||
      inputs.cache_dims.size() != inputs.expected_cache_rank ||
      inputs.value_dims.size() < 4 || inputs.cache_dims.size() < 4) {
    throw std::runtime_error("WebGPU update_cache: tensor rank changed");
  }
  for (int64_t dim : inputs.value_dims) {
    if (dim <= 0) {
      throw std::runtime_error(
          "WebGPU update_cache: value dimensions must be positive");
    }
  }
  for (int64_t dim : inputs.cache_dims) {
    if (dim <= 0) {
      throw std::runtime_error(
          "WebGPU update_cache: cache dimensions must be positive");
    }
  }

  const size_t value_rank = inputs.value_dims.size();
  const size_t cache_rank = inputs.cache_dims.size();
  if (inputs.value_dims[value_rank - 4] != 1 ||
      inputs.cache_dims[cache_rank - 4] != 1) {
    throw std::runtime_error("WebGPU update_cache: batch must be 1");
  }
  if (inputs.value_dims[value_rank - 2] !=
      inputs.cache_dims[cache_rank - 2]) {
    throw std::runtime_error("WebGPU update_cache: n_heads mismatch");
  }
  if (inputs.value_dims[value_rank - 1] !=
      inputs.cache_dims[cache_rank - 1]) {
    throw std::runtime_error("WebGPU update_cache: head_dim mismatch");
  }

  const uint64_t value_numel = utils::numel(inputs.value_dims);
  const uint64_t cache_numel = utils::numel(inputs.cache_dims);
  const uint64_t heads =
      static_cast<uint64_t>(inputs.value_dims[value_rank - 2]);
  const uint64_t head_dim =
      static_cast<uint64_t>(inputs.value_dims[value_rank - 1]);
  if (heads > std::numeric_limits<uint64_t>::max() / head_dim) {
    throw std::runtime_error("WebGPU update_cache: stride overflow");
  }
  const uint64_t stride = heads * head_dim;
  if (stride == 0) {
    throw std::runtime_error("WebGPU update_cache: stride must be positive");
  }
  if (inputs.start_pos < 0) {
    throw std::runtime_error(
        "WebGPU update_cache: input_pos must be non-negative");
  }
  const uint64_t start_pos = static_cast<uint64_t>(inputs.start_pos);
  if (start_pos > std::numeric_limits<uint64_t>::max() / stride) {
    throw std::runtime_error("WebGPU update_cache: input_pos offset overflow");
  }
  const uint64_t dst_offset = start_pos * stride;

  constexpr uint64_t kMaxU32 = std::numeric_limits<uint32_t>::max();
  if (cache_numel > kMaxU32 || value_numel > cache_numel ||
      value_numel > kMaxU32 || dst_offset > kMaxU32 ||
      dst_offset > cache_numel - value_numel) {
    throw std::runtime_error(
        "WebGPU update_cache: input_pos writes past cache capacity");
  }
  if (inputs.workgroup_size == 0 ||
      inputs.max_workgroups_per_dimension == 0) {
    throw std::runtime_error(
        "WebGPU update_cache: dispatch limits must be positive");
  }
  const uint64_t workgroup_count = value_numel / inputs.workgroup_size +
      static_cast<uint64_t>(value_numel % inputs.workgroup_size != 0);
  if (workgroup_count == 0 ||
      workgroup_count > inputs.max_workgroups_per_dimension) {
    throw std::runtime_error(
        "WebGPU update_cache: workgroup count exceeds the 1D dispatch limit");
  }

  LiveUpdateCacheState state = {};
  state.params.numel = static_cast<uint32_t>(value_numel);
  state.params.dst_offset = static_cast<uint32_t>(dst_offset);
  state.params.cache_numel = static_cast<uint32_t>(cache_numel);
  state.workgroup_count_x = static_cast<uint32_t>(workgroup_count);
  return state;
}

template <typename Commit>
void refresh_live_update_cache_state(
    const LiveUpdateCacheInputs& inputs,
    Commit&& commit) {
  const LiveUpdateCacheState state =
      compute_live_update_cache_state(inputs);
  commit(state);
}

} // namespace executorch::backends::webgpu
