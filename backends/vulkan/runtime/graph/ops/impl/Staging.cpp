/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/StagingUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/DimUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>

namespace vkcompute {

void add_staging_to_tensor_node(
    ComputeGraph& graph,
    const ValueRef in_staging,
    const ValueRef out_tensor) {
  vTensorPtr t_out = graph.get_tensor(out_tensor);
  VK_CHECK_COND(graph.val_is_staging(in_staging));

  api::ShaderInfo shader = get_nchw_to_image_shader(*t_out);

  api::utils::uvec3 global_size = t_out->extents();
  api::utils::uvec3 local_size = adaptive_work_group_size(global_size);

  graph.execute_nodes().emplace_back(new ExecuteNode(
      graph,
      shader,
      global_size,
      local_size,
      // Input and Outputs
      {{out_tensor, api::MemoryAccessType::WRITE},
       {in_staging, api::MemoryAccessType::READ}},
      // Parameter Buffers
      {t_out->sizes_ubo()},
      // Specialization Constants
      {SV(t_out->gpu_memory_layout_int())},
      // Resizing Logic
      nullptr,
      {}));
}

void add_tensor_to_staging_node(
    ComputeGraph& graph,
    const ValueRef in_tensor,
    const ValueRef out_staging) {
  vTensorPtr t_in = graph.get_tensor(in_tensor);
  VK_CHECK_COND(graph.val_is_staging(out_staging));

  api::ShaderInfo shader = get_image_to_nchw_shader(*t_in);

  api::utils::uvec3 global_size = t_in->extents();
  api::utils::uvec3 local_size = adaptive_work_group_size(global_size);

  graph.execute_nodes().emplace_back(new ExecuteNode(
      graph,
      shader,
      global_size,
      local_size,
      // Input and Outputs
      {{in_tensor, api::MemoryAccessType::READ},
       {out_staging, api::MemoryAccessType::WRITE}},
      // Parameter Buffers
      {t_in->sizes_ubo()},
      // Specialization Constants
      {SV(t_in->gpu_memory_layout_int())}));
}

ValueRef prepack(
    ComputeGraph& graph,
    const ValueRef vref,
    const api::GPUMemoryLayout layout) {
  ValueRef v = graph.add_tensor_like(vref, layout);
  vTensorPtr t = graph.get_tensor(v);

  api::ShaderInfo shader = get_nchw_to_image_shader(*t);

  api::utils::uvec3 global_size = t->extents();
  api::utils::uvec3 local_size = adaptive_work_group_size(global_size);

  graph.prepack_nodes().emplace_back(new PrepackNode(
      graph,
      shader,
      global_size,
      local_size,
      vref,
      v,
      {t->sizes_ubo()},
      // Specialization Constants
      {SV(t->gpu_memory_layout_int())}));

  return v;
}

ValueRef prepack_if_tensor_ref(
    ComputeGraph& graph,
    const ValueRef v,
    const api::GPUMemoryLayout layout) {
  if (graph.val_is_tref(v)) {
    return prepack(graph, v, layout);
  } else {
    return v;
  }
}

ValueRef prepack_if_tensor_ref(ComputeGraph& graph, const ValueRef v) {
  if (graph.val_is_tref(v)) {
    api::GPUMemoryLayout layout =
        graph.suggested_memory_layout(graph.get_tref(v)->sizes);
    return prepack(graph, v, layout);
  } else {
    return v;
  }
}

} // namespace vkcompute
