/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>

#include <executorch/backends/vulkan/runtime/graph/ops/DispatchNode.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>
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
    ubos.append({graph.sizes_ubo(out_tensor)});
  }

  graph.execute_nodes().emplace_back(new DispatchNode(
      graph,
      shader,
      graph.create_global_wg_size(out_tensor),
      graph.create_local_wg_size(out_tensor),
      // Input and Outputs
      {{out_tensor, vkapi::kWrite}, {in_staging, vkapi::kRead}},
      // Parameter Buffers
      ubos,
      // Specialization Constants
      {graph.hashed_layout_of(out_tensor)},
      // Resizing Logic
      nullptr,
      {}));
}

const std::string kBitw8PrefixStr = "bitw8_image_to_nchw_nobitw8buffer";

bool is_bitw8_shader(const vkapi::ShaderInfo& shader) {
  const auto size = kBitw8PrefixStr.size();
  const std::string& shader_prefix_str = shader.kernel_name.substr(0, size);
  return shader_prefix_str == kBitw8PrefixStr;
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
    ubos.append({graph.sizes_ubo(in_tensor)});
  }

  // Normally, the image_to_nchw shader is structured so that each thread reads
  // one texel from the input texture and writes each component of the texel
  // into the corresponding location in the output buffer. However, this shader
  // is structured slightly differently in that each thread writes out a
  // complete 32 bit integer (containing 4 packed 8-bit integers) into the
  // output buffer. Therefore, the global work group size for this shader will
  // be the number of elements in the output buffer divided by 4, as opposed to
  // the extents of the input texture.
  if (is_bitw8_shader(shader)) {
    uint32_t buffer_len = graph.get_staging(out_staging)->numel() / 4;
    global_wg_size = {buffer_len, 1, 1};
    ubos.append({graph.numel_ubo(in_tensor)});
  }

  graph.execute_nodes().emplace_back(new DispatchNode(
      graph,
      shader,
      global_wg_size,
      graph.create_local_wg_size(global_wg_size),
      // Input and Outputs
      {{out_staging, vkapi::kWrite}, {in_tensor, vkapi::kRead}},
      // Parameter Buffers
      ubos,
      // Specialization Constants
      {graph.hashed_layout_of(in_tensor)}));
}

void add_prepack_standard_node(
    ComputeGraph& graph,
    const ValueRef tensor_data,
    const ValueRef tensor) {
  vkapi::ShaderInfo shader = get_nchw_to_tensor_shader(
      *graph.get_tensor(tensor), graph.int8_buffers_enabled());

  vkapi::ParamsBindList ubos;
  if (graph.is_buffer_storage(tensor)) {
    ubos.append(
        {graph.sizes_ubo(tensor),
         graph.strides_ubo(tensor),
         graph.numel_ubo(tensor)});
  } else {
    ubos.append({graph.sizes_ubo(tensor)});
  }

  graph.prepack_nodes().emplace_back(new PrepackNode(
      graph,
      shader,
      graph.create_global_wg_size(tensor),
      graph.create_local_wg_size(tensor),
      // Input and Outputs
      tensor_data,
      tensor,
      // Parameter Buffers
      ubos,
      // Specialization Constants
      {graph.hashed_layout_of(tensor)}));
}

ValueRef prepack_standard(
    ComputeGraph& graph,
    const ValueRef tensor_data,
    const utils::StorageType storage_type,
    const utils::GPUMemoryLayout layout,
    const bool passthrough) {
  if (passthrough && graph.val_is_tensor(tensor_data)) {
    return tensor_data;
  }
  VK_CHECK_COND(graph.val_is_tref(tensor_data));
  ValueRef tensor = graph.add_tensor_like(tensor_data, storage_type, layout);
  add_prepack_standard_node(graph, tensor_data, tensor);
  return tensor;
}

ValueRef prepack_standard_like(
    ComputeGraph& graph,
    const ValueRef tensor_data,
    const ValueRef to_copy,
    const bool passthrough) {
  VK_CHECK_COND(graph.val_is_tensor(to_copy));
  return prepack_standard(
      graph,
      tensor_data,
      graph.storage_type_of(to_copy),
      graph.estimate_memory_layout_of(to_copy),
      passthrough);
}

void add_prepack_direct_copy_buffer_node(
    ComputeGraph& graph,
    const ValueRef tensor_data,
    const ValueRef tensor) {
  std::string kernel_name = "buffer_to_buffer";
  add_dtype_suffix(kernel_name, graph.dtype_of(tensor_data));
  vkapi::ShaderInfo shader = VK_KERNEL_FROM_STR(kernel_name);

  vkapi::ParamsBindList ubos;
  ubos.append({graph.numel_ubo(tensor)});

  graph.prepack_nodes().emplace_back(new PrepackNode(
      graph,
      shader,
      graph.create_global_wg_size(tensor),
      graph.create_local_wg_size(tensor),
      // Input and Outputs
      tensor_data,
      tensor,
      // Parameter Buffers
      ubos,
      // Specialization Constants
      {}));
}

ValueRef prepack_direct_copy_buffer(
    ComputeGraph& graph,
    const ValueRef tensor_data) {
  VK_CHECK_COND(graph.val_is_tref(tensor_data));
  ValueRef tensor =
      graph.add_tensor_like(tensor_data, utils::kBuffer, utils::kWidthPacked);
  add_prepack_direct_copy_buffer_node(graph, tensor_data, tensor);
  return tensor;
}

void prepack_op(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  return add_prepack_standard_node(graph, args[0], args[1]);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(et_vk.prepack.default, prepack_op);
}

} // namespace vkcompute
