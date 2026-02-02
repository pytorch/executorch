/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/Logging.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>

#include <executorch/backends/vulkan/runtime/graph/Logging.h>

namespace vkcompute {

//
// BlockConfig implementation
//

BlockConfig::BlockConfig(
    int32_t inner,
    int32_t inner_block_size,
    int32_t outer,
    int32_t outer_block_size,
    bool transposed)
    : inner_dim(inner),
      inner_dim_block_size(inner_block_size),
      outer_dim(outer),
      outer_dim_block_size(outer_block_size),
      block_transposed(transposed),
      block_dim_order{0, 0, 0, 0} {
  // Block dims must be different
  VK_CHECK_COND(outer_dim != inner_dim);

  // Find the two lowest dim indices that are not inner_dim or outer_dim
  int32_t first_nonblock_dim = -1;
  int32_t second_nonblock_dim = -1;
  int32_t other_idx = 0;
  for (int32_t d = 0; other_idx < 2; ++d) {
    if (d != inner_dim && d != outer_dim) {
      if (other_idx == 0) {
        first_nonblock_dim = d;
      } else {
        second_nonblock_dim = d;
      }
      ++other_idx;
    }
  }

  // Set block_dim_order based on block_transposed
  if (block_transposed) {
    // Transposed: {outer_dim, inner_dim, first_nonblock_dim,
    // second_nonblock_dim}
    block_dim_order[0] = outer_dim;
    block_dim_order[1] = inner_dim;
  } else {
    // Normal: {inner_dim, outer_dim, first_nonblock_dim, second_nonblock_dim}
    block_dim_order[0] = inner_dim;
    block_dim_order[1] = outer_dim;
  }
  block_dim_order[2] = first_nonblock_dim;
  block_dim_order[3] = second_nonblock_dim;

  // Validate all dims are in valid range [0, 3]
  for (int i = 0; i < 4; ++i) {
    VK_CHECK_COND(block_dim_order[i] >= 0 && block_dim_order[i] < 4);
  }
}

int32_t BlockConfig::as_packed_int() const {
  int32_t packed = 0;
  // Pack block_dim_order in bits 0-15 (matches hashed layout format)
  packed |= (block_dim_order[0] & 0xF); // bits 0-3
  packed |= (block_dim_order[1] & 0xF) << 4; // bits 4-7
  packed |= (block_dim_order[2] & 0xF) << 8; // bits 8-11
  packed |= (block_dim_order[3] & 0xF) << 12; // bits 12-15
  // Pack packed_dim_info in bits 16-31 (matches hashed layout format)
  packed |= (inner_dim & 0xF) << 16; // bits 16-19
  packed |= (outer_dim & 0xF) << 20; // bits 20-23
  packed |= (inner_dim_block_size & 0xF) << 24; // bits 24-27
  packed |= (outer_dim_block_size & 0xF) << 28; // bits 28-31

  return packed;
}

int32_t BlockConfig::inner_dim_from_packed_int(int32_t packed_int) {
  return (packed_int >> 16) & 0xF; // bits 16-19
}

int32_t BlockConfig::outer_dim_from_packed_int(int32_t packed_int) {
  return (packed_int >> 20) & 0xF; // bits 20-23
}

//
// Default workgroup size functions
//

utils::uvec3 default_pick_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;
  const ValueRef out = args.at(0).refs.at(0);
  return graph->create_global_wg_size(out);
}

utils::uvec3 default_pick_local_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)args;
  (void)resize_args;
  return graph->create_local_wg_size(global_workgroup_size);
}

utils::uvec3 pick_hw_square_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)graph;
  (void)shader;
  (void)args;
  (void)resize_args;
  // Some inactive invocations are okay; set 6 as the threshold to use the
  // a square wg size.
  if (global_workgroup_size[0u] >= 6 && global_workgroup_size[1u] >= 6) {
    return {8u, 8u, 1u};
  }
  // If width dim is sufficiently small, then bias towards height dim to reduce
  // the number of inactive invocations.
  if (global_workgroup_size[0u] < 6u) {
    return {4u, 16u, 1u};
  }
  return {16u, 4u, 1u};
}

utils::uvec3 pick_wc_square_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)graph;
  (void)shader;
  (void)args;
  (void)resize_args;
  // Some inactive invocations are okay; set 6 as the threshold to use the
  // a square wg size.
  if (global_workgroup_size[0u] >= 6 && global_workgroup_size[2u] >= 6) {
    return {8u, 1u, 8u};
  }
  // If channels dim is sufficiently small, then bias towards width dim to
  // reduce the number of inactive invocations.
  if (global_workgroup_size[2u] < 2u) {
    return {64u, 1u, 1u};
  }
  return {16u, 1u, 4u};
}

BlockConfig create_block_config_from_io_packed_dims(
    ComputeGraph& graph,
    const ValueRef output,
    const ValueRef input) {
  const int32_t block_inner_dim = graph.packed_dim_of(output);
  int32_t block_outer_dim = graph.packed_dim_of(input);

  // If inner and outer dims are the same, pick a different outer dim
  if (block_outer_dim == block_inner_dim) {
    if (block_inner_dim == 0) {
      block_outer_dim = 1;
    } else {
      block_outer_dim = 0;
    }
  }

  // Create a BlockConfig with block sizes of 4 for both dimensions
  return BlockConfig{block_inner_dim, 4, block_outer_dim, 4};
}

BlockConfig create_block_config_for_tensor(
    ComputeGraph& graph,
    const ValueRef tensor) {
  const int32_t packed_dim = graph.packed_dim_of(tensor);

  // Pick an outer dimension that differs from the packed dimension
  const int32_t outer_dim = (packed_dim == 0) ? 1 : 0;

  // Create a BlockConfig with block sizes of 4 for both dimensions
  return BlockConfig{packed_dim, 4, outer_dim, 4};
}

BlockConfig create_block_config_from_other(
    ComputeGraph& graph,
    const ValueRef tensor,
    const BlockConfig& other) {
  const int32_t packed_dim = graph.packed_dim_of(tensor);

  // If tensor's packed dim matches other's inner dim, use same config
  if (packed_dim == other.inner_dim) {
    return other;
  }

  // Otherwise, transpose: swap inner and outer dimensions
  return BlockConfig{
      other.outer_dim,
      other.outer_dim_block_size,
      other.inner_dim,
      other.inner_dim_block_size};
}

utils::uvec3 pick_linear_global_wg_with_block_config(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  (void)shader;

  const ValueRef output = args.at(0).refs.at(0);
  // extra_args contains the packed block config directly as a ValueRef
  // (int32_t)
  const int32_t packed_block_config = static_cast<int32_t>(extra_args.at(0));

  // Extract block configuration from packed integer
  const int32_t inner_dim =
      BlockConfig::inner_dim_from_packed_int(packed_block_config);
  const int32_t outer_dim =
      BlockConfig::outer_dim_from_packed_int(packed_block_config);

  const std::vector<int64_t>& sizes = graph->sizes_of(output);
  const size_t ndim = sizes.size();

  // Compute number of blocks along inner and outer dimensions
  const int64_t inner_size = sizes[ndim - 1 - inner_dim];
  const int64_t outer_size = sizes[ndim - 1 - outer_dim];

  const uint32_t num_inner_blocks =
      utils::safe_downcast<uint32_t>(utils::div_up(inner_size, int64_t(4)));
  const uint32_t num_outer_blocks =
      utils::safe_downcast<uint32_t>(utils::div_up(outer_size, int64_t(4)));

  // Compute number of planes (product of dimensions not in the block)
  uint32_t num_planes = 1;
  for (size_t i = 0; i < ndim; ++i) {
    const int32_t whcn_dim = ndim - 1 - i;
    if (whcn_dim != inner_dim && whcn_dim != outer_dim) {
      num_planes *= utils::safe_downcast<uint32_t>(sizes[i]);
    }
  }

  // Return linear workgroup size: {total_blocks, 1u, 1u}
  const uint32_t total_blocks =
      num_inner_blocks * num_outer_blocks * num_planes;
  return {total_blocks, 1u, 1u};
}

utils::uvec3 pick_extents_global_wg_with_block_config(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  (void)shader;

  const ValueRef output = args.at(0).refs.at(0);
  // extra_args contains the packed block config directly as a ValueRef
  // (int32_t)
  const int32_t packed_block_config = static_cast<int32_t>(extra_args.at(0));

  // Extract block configuration from packed integer
  // Note: inner_dim and outer_dim use WHCN order (0=W, 1=H, 2=C, 3=N)
  const int32_t inner_dim =
      BlockConfig::inner_dim_from_packed_int(packed_block_config);
  const int32_t outer_dim =
      BlockConfig::outer_dim_from_packed_int(packed_block_config);

  const std::vector<int64_t>& sizes = graph->sizes_of(output);

  // C++ sizes are in NCHW order: sizes[0]=N, sizes[1]=C, sizes[2]=H, sizes[3]=W
  // Access dimensions from the end for tensors with fewer than 4 dims
  const int64_t W = utils::val_at(-1, sizes);
  const int64_t H = utils::val_at(-2, sizes);
  const int64_t C = utils::val_at(-3, sizes);
  const int64_t N = utils::val_at(-4, sizes);

  // Dispatch structure: {x_threads, y_threads, z_threads}
  // - x corresponds to W dimension
  // - y corresponds to H dimension
  // - z corresponds to C * N (combined)
  //
  // Block dimensions (inner_dim and outer_dim) are divided by 4,
  // non-block dimensions are not divided.

  uint32_t x_threads, y_threads;
  int64_t C_for_z;

  // X dimension (W, WHCN dim 0)
  if (inner_dim == 0 || outer_dim == 0) {
    x_threads = utils::safe_downcast<uint32_t>(utils::div_up(W, int64_t(4)));
  } else {
    x_threads = utils::safe_downcast<uint32_t>(W);
  }

  // Y dimension (H, WHCN dim 1)
  if (inner_dim == 1 || outer_dim == 1) {
    y_threads = utils::safe_downcast<uint32_t>(utils::div_up(H, int64_t(4)));
  } else {
    y_threads = utils::safe_downcast<uint32_t>(H);
  }

  // Z dimension: C * N where C is blocked if it's part of the block
  if (inner_dim == 2 || outer_dim == 2) {
    C_for_z = utils::div_up(C, int64_t(4));
  } else {
    C_for_z = C;
  }
  const uint32_t z_threads = utils::safe_downcast<uint32_t>(C_for_z * N);

  return {x_threads, y_threads, z_threads};
}

utils::uvec3 pick_square_local_wg_with_block_config(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  (void)graph;
  (void)shader;
  (void)args;

  // Detect linear dispatch pattern: global_wg = {total_blocks, 1, 1}
  if (global_workgroup_size[1u] == 1u && global_workgroup_size[2u] == 1u) {
    return {64u, 1u, 1u};
  }

  // Extents dispatch: use 8x8 square on inner_dim and outer_dim axes
  // extra_args contains the packed block config as a ValueRef (int32_t)
  const int32_t packed_block_config = static_cast<int32_t>(extra_args.at(0));

  // Extract block configuration from packed integer
  // inner_dim and outer_dim use WHCN order (0=W, 1=H, 2=C, 3=N)
  const int32_t inner_dim =
      BlockConfig::inner_dim_from_packed_int(packed_block_config);
  const int32_t outer_dim =
      BlockConfig::outer_dim_from_packed_int(packed_block_config);

  // Build local workgroup size:
  // - x corresponds to W (WHCN dim 0)
  // - y corresponds to H (WHCN dim 1)
  // - z corresponds to C*N (WHCN dim 2 for C)
  // Set axes in the block (inner_dim, outer_dim) to 8, others to 1
  uint32_t local_x = (inner_dim == 0 || outer_dim == 0) ? 8u : 1u;
  uint32_t local_y = (inner_dim == 1 || outer_dim == 1) ? 8u : 1u;
  uint32_t local_z = (inner_dim == 2 || outer_dim == 2) ? 8u : 1u;

  return {local_x, local_y, local_z};
}

} // namespace vkcompute
