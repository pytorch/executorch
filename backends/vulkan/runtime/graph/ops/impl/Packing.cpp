/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/api/Utils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Packing.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

ValueRef channel_image_repacking(
    ComputeGraph& graph,
    ValueRef in,
    api::GPUMemoryLayout target_layout,
    std::string kernel_name) {
  const auto sizes = graph.get_sizes_of(in);

  ValueRef out = graph.add_tensor(
      sizes, graph.get_dtype_of(in), api::kTexture3D, target_layout);
  vTensorPtr t = graph.get_tensor(out);

  api::utils::uvec3 global_size = t->extents();
  api::utils::uvec3 local_size = adaptive_work_group_size(global_size);

  add_dtype_suffix(kernel_name, *t);

  graph.execute_nodes().emplace_back(new ExecuteNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      global_size,
      local_size,
      // Inputs and Outputs
      {{out, api::MemoryAccessType::WRITE}, {in, api::MemoryAccessType::READ}},
      // Shader params buffers
      {
          // The shader assumes a 4d nchw to calculate the lookup coordinate.
          // If the input is not 4d, we need to pad it with 1's on the front.
          graph.create_params_buffer(
              api::utils::make_ivec4_prepadded1(graph.get_sizes_of(in))),
          t->texture_limits_ubo(),
      },
      // Specialization constants
      {}));

  return out;
}

ValueRef convert_image_channels_packed_to_width_packed(
    ComputeGraph& graph,
    ValueRef in) {
  std::string kernel_name("convert_channels_to_width_packed");
  kernel_name.reserve(kShaderNameReserve);
  return channel_image_repacking(graph, in, api::kWidthPacked, kernel_name);
}

ValueRef convert_image_channels_packed_to_height_packed(
    ComputeGraph& graph,
    ValueRef in) {
  std::string kernel_name("convert_channels_to_height_packed");
  kernel_name.reserve(kShaderNameReserve);
  return channel_image_repacking(graph, in, api::kHeightPacked, kernel_name);
}

} // namespace vkcompute
