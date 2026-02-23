/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>
#include <executorch/backends/vulkan/runtime/graph/ops/ExecuteNode.h>

namespace vkcompute {

/**
 * BlockConfig describes how a tensor is partitioned into blocks for the purpose
 * of thread mapping in GPU compute shaders. Each thread processes one block
 * of elements.
 *
 * This is distinct from PackedDimInfo in Tensor.h which describes memory
 * layout. BlockConfig is used solely for operator implementations to define 4x4
 * block partitioning schemes.
 *
 * The block configuration has two dimensions:
 *   - inner_dim: The dimension where 4 consecutive elements are processed
 *     together within a single thread
 *   - outer_dim: A second dimension where 4 elements are grouped, resulting
 *     in a 4x4 block of 16 elements per thread
 */
struct BlockConfig {
  // The inner block dimension (WHCN index: 0=W, 1=H, 2=C, 3=N)
  // 4 consecutive elements along this dimension form the inner part of a block
  int32_t inner_dim;
  // Block size along the inner dimension (typically 4)
  int32_t inner_dim_block_size;
  // The outer block dimension (WHCN index: 0=W, 1=H, 2=C, 3=N)
  // 4 elements along this dimension form the outer part of a block
  int32_t outer_dim;
  // Block size along the outer dimension (typically 4)
  int32_t outer_dim_block_size;
  // Whether the block is transposed (swaps stride ordering of inner/outer dim)
  bool block_transposed;
  // Dimension order for the block:
  // - If block_transposed = false: {inner_dim, outer_dim, first_nonblock_dim,
  //   second_nonblock_dim}
  // - If block_transposed = true: {outer_dim, inner_dim, first_nonblock_dim,
  //   second_nonblock_dim}
  int32_t block_dim_order[4];

  BlockConfig(
      int32_t inner,
      int32_t inner_block_size,
      int32_t outer,
      int32_t outer_block_size,
      bool transposed = false);

  /**
   * Returns a packed int32_t encoding the block configuration. The structure
   * matches the hashed layout int format used in shaders:
   *   bits  0- 3: block_dim_order[0]
   *   bits  4- 7: block_dim_order[1]
   *   bits  8-11: block_dim_order[2]
   *   bits 12-15: block_dim_order[3]
   *   bits 16-19: inner_dim
   *   bits 20-23: outer_dim
   *   bits 24-27: inner_dim_block_size
   *   bits 28-31: outer_dim_block_size
   */
  int32_t as_packed_int() const;

  /**
   * Extracts inner_dim from a packed int32_t representation.
   */
  static int32_t inner_dim_from_packed_int(int32_t packed_int);

  /**
   * Extracts outer_dim from a packed int32_t representation.
   */
  static int32_t outer_dim_from_packed_int(int32_t packed_int);
};

/**
 * Creates a global workgroup size based on the first output tensor in the args.
 * This is a utility function that extracts the output tensor from
 * args.at(0).refs.at(0) and calls graph->create_global_wg_size(out) on it.
 */
utils::uvec3 default_pick_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args);

/**
 * Creates a local workgroup size based on the first output tensor in the args.
 * This is a utility function that extracts the output tensor from
 * args.at(0).refs.at(0) and calls graph->create_local_wg_size(out) on it.
 */
utils::uvec3 default_pick_local_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args);

/**
 * Constructs a local work group size with the shape {W, H, 1}. The function
 * will try to set W == H == sqrt(num_invocations), where num_invocations is
 * typically 64. This configuration is good for ops like matrix multiplication
 * as it reduces the total volume of unique data that the entire work group
 * will need to read from input tensors in order to produce the output data.
 * To compute an output tile of {W, H, 1}, the work group will need to read
 * H unique rows = H * K unique elements from the input tensor and W unique cols
 * = W * K elements from the weight tensor, resulting in (W + H) * K unique
 * elements in total.
 */
utils::uvec3 pick_hw_square_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args);

utils::uvec3 pick_wc_square_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args);

/**
 * Creates a BlockConfig based on the packed dimensions of an output and input
 * tensor pair. This is useful for operations like dequantize where the block
 * configuration depends on both tensors.
 *
 * The inner dimension is determined by the output tensor's packed dimension,
 * and the outer dimension is determined by the input tensor's packed dimension.
 * If they are the same, the outer dimension is adjusted to avoid conflict.
 *
 * @param graph The compute graph
 * @param output The output tensor reference
 * @param input The input tensor reference
 * @return A BlockConfig configured for block-based operations
 */
BlockConfig create_block_config_from_io_packed_dims(
    ComputeGraph& graph,
    const ValueRef output,
    const ValueRef input);

/**
 * Creates a BlockConfig based on the packed dimension of a single tensor.
 * This is useful when you need separate block configs for input and output
 * tensors.
 *
 * The inner dimension is determined by the tensor's packed dimension.
 * The outer dimension is set to an adjacent dimension that differs from
 * the packed dimension.
 *
 * @param graph The compute graph
 * @param tensor The tensor reference
 * @return A BlockConfig configured for block-based operations
 */
BlockConfig create_block_config_for_tensor(
    ComputeGraph& graph,
    const ValueRef tensor);

/**
 * Creates a BlockConfig for a tensor based on another block config, ensuring
 * the inner dimension matches the tensor's packed dimension.
 *
 * This is useful when you need block configs for both input and output tensors
 * that share the same block axes but may need to be transposed if the tensors
 * have different packed dimensions.
 *
 * If the tensor's packed dim matches the other config's inner dim, returns
 * the same config. Otherwise, returns a transposed config (inner/outer
 * swapped).
 *
 * @param graph The compute graph
 * @param tensor The tensor to create a block config for
 * @param other The reference block config to base the new config on
 * @return A BlockConfig with inner_dim = tensor's packed_dim
 */
BlockConfig create_block_config_from_other(
    ComputeGraph& graph,
    const ValueRef tensor,
    const BlockConfig& other);

/**
 * Picks a global workgroup size for block-based dispatching using a linear
 * (1D flattened) dispatch pattern. This is optimized for buffer storage.
 *
 * This function expects:
 *   - args.at(0).refs.at(0): Output tensor reference
 *   - extra_args.at(0): Packed int32_t block configuration cast to ValueRef
 *     (created via static_cast<ValueRef>(BlockConfig::as_packed_int()))
 *
 * The global workgroup size is computed as:
 *   - x = total_blocks = num_inner_blocks * num_outer_blocks * num_planes
 *   - y = 1
 *   - z = 1
 *
 * @return Global workgroup size as {total_blocks, 1, 1}
 */
utils::uvec3 pick_linear_global_wg_with_block_config(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args);

/**
 * Picks a global workgroup size for block-based dispatching using a 3D
 * extents-style dispatch pattern. This is optimized for texture storage.
 *
 * This function expects:
 *   - args.at(0).refs.at(0): Output tensor reference
 *   - extra_args.at(0): Packed int32_t block configuration cast to ValueRef
 *     (created via static_cast<ValueRef>(BlockConfig::as_packed_int()))
 *
 * The global workgroup size is computed as a WHCN-based 3D dispatch:
 *   - x = W threads (divided by 4 if W is inner or outer dim)
 *   - y = H threads (divided by 4 if H is inner or outer dim)
 *   - z = C * N threads (C divided by 4 if C is inner or outer dim)
 *
 * @return Global workgroup size as {x_threads, y_threads, z_threads}
 */
utils::uvec3 pick_extents_global_wg_with_block_config(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args);

/**
 * Picks a local workgroup size for block-based dispatching that is optimized
 * for the dispatch pattern in use.
 *
 * This function expects:
 *   - extra_args.at(0): Packed int32_t block configuration cast to ValueRef
 *     (created via static_cast<ValueRef>(BlockConfig::as_packed_int()))
 *
 * For linear dispatch (buffer storage, global_wg = {total_blocks, 1, 1}):
 *   - Returns {64, 1, 1}
 *
 * For extents dispatch (texture storage, global_wg = {x, y, z}):
 *   - Returns an 8x8 square configuration where:
 *     - Axes corresponding to inner_dim and outer_dim are set to 8
 *     - The remaining axis is set to 1
 *   - For example: inner_dim=W, outer_dim=H -> {8, 8, 1}
 *                  inner_dim=W, outer_dim=C -> {8, 1, 8}
 *
 * @return Local workgroup size optimized for the dispatch pattern
 */
utils::uvec3 pick_square_local_wg_with_block_config(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args);

} // namespace vkcompute
