/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/api/api.h>
#include <executorch/backends/vulkan/runtime/graph/Logging.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/KernelUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

#include <iostream>

namespace vkcompute {

void add_copy_offset_node(
    ComputeGraph& graph,
    const ValueRef in,
    const api::utils::ivec3& range,
    const api::utils::ivec3& src_offset,
    const api::utils::ivec3& dst_offset,
    const ValueRef out) {
  vTensorPtr t_in = graph.get_tensor(in);
  vTensorPtr t_out = graph.get_tensor(out);

  VK_CHECK_COND(check_memory_layout_is(*t_in, api::kChannelsPacked));
  VK_CHECK_COND(check_memory_layout_is(*t_out, api::kChannelsPacked));

  std::string kernel_name = "copy_offset";
  kernel_name.reserve(kShaderNameReserve);
  add_dtype_suffix(kernel_name, *t_out);

  api::utils::uvec3 global_size = api::utils::make_uvec3(range);
  api::utils::uvec3 local_size = adaptive_work_group_size(global_size);

  const struct Block final {
    api::utils::ivec3 range;
    int32_t unused0;
    api::utils::ivec3 src_offset;
    int32_t unused1;
    api::utils::ivec3 dst_offset;
    int32_t unused2;
  } offset_params{
      range,
      0,
      src_offset,
      0,
      dst_offset,
      0,
  };

  auto shader = VK_KERNEL_FROM_STR(kernel_name);

  graph.execute_nodes().emplace_back(new ExecuteNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      global_size,
      local_size,
      // Inputs and Outputs
      {{out, api::MemoryAccessType::WRITE}, {in, api::MemoryAccessType::READ}},
      // Parameter buffers
      {t_out->texture_limits_ubo(),
       t_in->texture_limits_ubo(),
       graph.create_params_buffer(offset_params)},
      // Specialization Constants
      {}));
}

} // namespace vkcompute
