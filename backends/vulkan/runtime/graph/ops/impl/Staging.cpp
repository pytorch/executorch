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

  vkapi::ShaderInfo shader = get_nchw_to_tensor_shader(
      *graph.get_tensor(out_tensor), graph.int8_buffers_enabled());

  vkapi::ParamsBindList ubos;
  if (graph.is_buffer_storage(out_tensor)) {
    ubos.append(
        {graph.sizes_ubo(out_tensor),
         graph.strides_ubo(out_tensor),
         graph.numel_ubo(out_tensor)});
  } else {
    ubos.append({graph.sizes_ubo(out_tensor), graph.axis_map_ubo(out_tensor)});
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

  vkapi::ShaderInfo shader = get_tensor_to_nchw_shader(
      *graph.get_tensor(in_tensor), graph.int8_buffers_enabled());

  utils::uvec3 global_wg_size = graph.create_global_wg_size(in_tensor);

  vkapi::ParamsBindList ubos;
  if (graph.is_buffer_storage(in_tensor)) {
    ubos.append(
        {graph.sizes_ubo(in_tensor),
         graph.strides_ubo(in_tensor),
         graph.numel_ubo(in_tensor)});
  } else {
    ubos.append({graph.sizes_ubo(in_tensor), graph.axis_map_ubo(in_tensor)});
  }

  // Normally, the image_to_nchw shader is structured so that each thread reads
  // one texel from the input texture and writes each component of the texel
  // into the corresponding location in the output buffer. However, this shader
  // is structured slightly differently in that each thread writes out a
  // complete 32 bit integer (containing 4 packed 8-bit integers) into the
  // output buffer. Therefore, the global work group size for this shader will
  // be the number of elements in the output buffer divided by 4, as opposed to
  // the extents of the input texture.
  if (shader.kernel_name == "int8_image_to_nchw_noint8") {
    uint32_t buffer_len = graph.get_staging(out_staging)->numel() / 4;
    global_wg_size = {buffer_len, 1, 1};
    ubos.append({graph.numel_ubo(in_tensor)});
  }

  graph.execute_nodes().emplace_back(new ExecuteNode(
      graph,
      shader,
      global_wg_size,
      graph.create_local_wg_size(global_wg_size),
      // Input and Outputs
      {{out_staging, vkapi::MemoryAccessType::WRITE},
       {in_tensor, vkapi::MemoryAccessType::READ}},
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

  vkapi::ShaderInfo shader = get_nchw_to_tensor_shader(
      *graph.get_tensor(v), graph.int8_buffers_enabled());

  vkapi::ParamsBindList ubos;
  if (graph.is_buffer_storage(v)) {
    ubos.append({graph.sizes_ubo(v), graph.strides_ubo(v), graph.numel_ubo(v)});
  } else {
    ubos.append({graph.sizes_ubo(v), graph.axis_map_ubo(v)});
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

ValueRef prepack_buffer(
    ComputeGraph& graph,
    const ValueRef vref,
    const utils::GPUMemoryLayout layout) {
  ValueRef v = graph.add_tensor_like(vref, utils::kBuffer, layout);

  vkapi::ShaderInfo shader = VK_KERNEL_FROM_STR("buffer_to_buffer");

  vkapi::ParamsBindList ubos;
  ubos.append({graph.numel_ubo(v)});

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
      {}));

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

ValueRef prepack_buffer_if_tensor_ref(
    ComputeGraph& graph,
    const ValueRef v,
    const utils::GPUMemoryLayout layout) {
  if (graph.val_is_tref(v)) {
    return prepack_buffer(graph, v, layout);
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

ValueRef prepack_buffer_if_tensor_ref(ComputeGraph& graph, const ValueRef v) {
  if (graph.val_is_tref(v)) {
    utils::GPUMemoryLayout layout =
        graph.suggested_memory_layout(graph.get_tref(v)->sizes);
    return prepack_buffer(graph, v, layout);
  } else {
    return v;
  }
}

} // namespace vkcompute
