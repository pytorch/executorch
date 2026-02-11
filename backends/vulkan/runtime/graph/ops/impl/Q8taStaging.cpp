/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Q8taStaging.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

void add_staging_to_int8x4_buffer_node(
    ComputeGraph& graph,
    const ValueRef tensor_data,
    const ValueRef tensor) {
  VK_CHECK_COND(graph.dtype_of(tensor) == vkapi::kInt8x4);

  std::string kernel_name = "nchw_to_int8x4_buffer";

  vkapi::ParamsBindList param_buffers;
  param_buffers.append(graph.buffer_meta_ubo(tensor));

  // One thread per texel (each texel = one int32 = 4 packed int8)
  uint32_t num_texels =
      utils::safe_downcast<uint32_t>(graph.numel_of(tensor) / 4);
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

} // namespace vkcompute
