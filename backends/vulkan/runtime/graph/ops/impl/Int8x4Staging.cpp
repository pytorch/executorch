/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Int8x4Staging.h>

#include <executorch/backends/vulkan/runtime/graph/ops/DynamicDispatchNode.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

void add_prepack_int8x4_buffer_node(
    ComputeGraph& graph,
    const ValueRef tensor_data,
    const ValueRef tensor) {
  VK_CHECK_COND(graph.dtype_of(tensor) == vkapi::kInt8x4);
  // TODO(ssjia): Update shaders to handle high-dim tensors
  VK_CHECK_COND(graph.dim_of(tensor) <= 4);

  std::string kernel_name = "nchw_to_int8x4_buffer";

  vkapi::ParamsBindList param_buffers;
  param_buffers.append(graph.buffer_meta_ubo(tensor));

  // One thread per texel (each texel = one int32 = 4 packed int8).
  // Use padded_numel to account for dimension padding in packed int8 layouts
  // (e.g., kPackedInt8_4C with C=3 pads to C=4).
  uint32_t num_texels =
      utils::safe_downcast<uint32_t>(graph.padded_numel_of(tensor) / 4);
  utils::uvec3 global_wg_size = {num_texels, 1, 1};
  utils::uvec3 local_wg_size = graph.create_local_wg_size(global_wg_size);

  graph.prepack_nodes().emplace_back(new PrepackNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      global_wg_size,
      local_wg_size,
      // Input and Output
      tensor_data,
      tensor,
      // Parameter Buffers
      param_buffers,
      // Specialization Constants
      {graph.hashed_layout_of(tensor)}));
}

static utils::uvec3 staging_to_int8x4_buffer_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;
  const ValueRef out_tensor = args.at(0).refs.at(0);
  const uint32_t num_texels =
      utils::safe_downcast<uint32_t>(graph->padded_numel_of(out_tensor) / 4);
  return {num_texels, 1, 1};
}

void add_staging_to_int8x4_buffer_node(
    ComputeGraph& graph,
    const ValueRef in_staging,
    const ValueRef tensor) {
  VK_CHECK_COND(graph.dtype_of(tensor) == vkapi::kInt8x4);
  // TODO(ssjia): Update shaders to handle high-dim tensors
  VK_CHECK_COND(graph.dim_of(tensor) <= 4);

  vkapi::ParamsBindList param_buffers;
  param_buffers.append(graph.buffer_meta_ubo(tensor));

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR("nchw_to_int8x4_buffer"),
      staging_to_int8x4_buffer_global_wg_size,
      default_pick_local_wg_size,
      // Input and Output
      {{tensor, vkapi::kWrite}, {in_staging, vkapi::kRead}},
      // Parameter Buffers
      param_buffers,
      // Push Constants
      {},
      // Specialization Constants
      {graph.hashed_layout_of(tensor)},
      // Resize Args
      {},
      // Resizing Logic
      nullptr));
}

static utils::uvec3 int8x4_buffer_to_staging_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;
  const ValueRef in_tensor = args.at(1).refs.at(0);
  // One thread per output int32 in the NCHW staging buffer.
  const int32_t numel = graph->numel_of(in_tensor);
  const uint32_t num_out_int32s =
      utils::safe_downcast<uint32_t>((numel + 3) / 4);
  return {num_out_int32s, 1, 1};
}

void add_int8x4_buffer_to_staging_node(
    ComputeGraph& graph,
    const ValueRef tensor,
    const ValueRef staging_data) {
  VK_CHECK_COND(graph.dtype_of(tensor) == vkapi::kInt8x4);
  // TODO(ssjia): Update shaders to handle high-dim tensors
  VK_CHECK_COND(graph.dim_of(tensor) <= 4);

  vkapi::ParamsBindList param_buffers;
  param_buffers.append(graph.buffer_meta_ubo(tensor));

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR("int8x4_buffer_to_nchw"),
      int8x4_buffer_to_staging_global_wg_size,
      default_pick_local_wg_size,
      // Input and Output
      {{staging_data, vkapi::kWrite}, {tensor, vkapi::kRead}},
      // Parameter Buffers
      param_buffers,
      // Push Constants
      {},
      // Specialization Constants
      {graph.hashed_layout_of(tensor)},
      // Resize Args
      {},
      // Resizing Logic
      nullptr));
}

} // namespace vkcompute
