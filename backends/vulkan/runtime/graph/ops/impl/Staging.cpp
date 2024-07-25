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
  VK_CHECK_COND(graph.val_is_staging(in_staging));

  vkapi::ShaderInfo shader =
      get_nchw_to_tensor_shader(*graph.get_tensor(out_tensor));

  vkapi::ParamsBindList ubos({graph.sizes_ubo(out_tensor)});
  if (graph.is_buffer_storage(out_tensor)) {
    ubos.append({
        graph.texel_strides_ubo(out_tensor),
        graph.ntexels_ubo(out_tensor),
    });
  }

  graph.execute_nodes().emplace_back(new ExecuteNode(
      graph,
      shader,
      graph.create_global_wg_size(out_tensor),
      graph.create_local_wg_size(out_tensor),
      // Input and Outputs
      {{out_tensor, vkapi::MemoryAccessType::WRITE},
       {in_staging, vkapi::MemoryAccessType::READ}},
      // Parameter Buffers
      ubos,
      // Specialization Constants
      {SV(graph.packed_dim_whcn_idx_of(out_tensor))},
      // Resizing Logic
      nullptr,
      {}));
}

void add_tensor_to_staging_node(
    ComputeGraph& graph,
    const ValueRef in_tensor,
    const ValueRef out_staging) {
  VK_CHECK_COND(graph.val_is_staging(out_staging));

  vkapi::ShaderInfo shader =
      get_tensor_to_nchw_shader(*graph.get_tensor(in_tensor));

  vkapi::ParamsBindList ubos({graph.sizes_ubo(in_tensor)});
  if (graph.is_buffer_storage(in_tensor)) {
    ubos.append({
        graph.texel_strides_ubo(in_tensor),
        graph.ntexels_ubo(in_tensor),
    });
  }

  graph.execute_nodes().emplace_back(new ExecuteNode(
      graph,
      shader,
      graph.create_global_wg_size(in_tensor),
      graph.create_local_wg_size(in_tensor),
      // Input and Outputs
      {{in_tensor, vkapi::MemoryAccessType::READ},
       {out_staging, vkapi::MemoryAccessType::WRITE}},
      // Parameter Buffers
      ubos,
      // Specialization Constants
      {SV(graph.packed_dim_whcn_idx_of(in_tensor))}));
}

ValueRef prepack(
    ComputeGraph& graph,
    const ValueRef vref,
    const utils::GPUMemoryLayout layout) {
  ValueRef v = graph.add_tensor_like(vref, layout);

  vkapi::ShaderInfo shader = get_nchw_to_tensor_shader(*graph.get_tensor(v));

  vkapi::ParamsBindList ubos({graph.sizes_ubo(v)});
  if (graph.is_buffer_storage(v)) {
    ubos.append({
        graph.texel_strides_ubo(v),
        graph.ntexels_ubo(v),
    });
  }

  graph.prepack_nodes().emplace_back(new PrepackNode(
      graph,
      shader,
      graph.create_global_wg_size(v),
      graph.create_local_wg_size(v),
      // Input and Outputs
      vref,
      v,
      // Parameter Buffers
      ubos,
      // Specialization Constants
      {SV(graph.packed_dim_whcn_idx_of(v))}));

  return v;
}

ValueRef prepack_if_tensor_ref(
    ComputeGraph& graph,
    const ValueRef v,
    const utils::GPUMemoryLayout layout) {
  if (graph.val_is_tref(v)) {
    return prepack(graph, v, layout);
  } else {
    return v;
  }
}

ValueRef prepack_if_tensor_ref(ComputeGraph& graph, const ValueRef v) {
  if (graph.val_is_tref(v)) {
    utils::GPUMemoryLayout layout =
        graph.suggested_memory_layout(graph.get_tref(v)->sizes);
    return prepack(graph, v, layout);
  } else {
    return v;
  }
}

} // namespace vkcompute
